"""Persistent, independently seeded training sessions with optional processes."""

import random
import signal
import traceback
from multiprocessing import get_context

from .config import task_for
from .recording import EpisodeLedger
from .session import Session


class MemoryRecorder:
    steps = agents = None

    def __init__(self):
        self.episodes = []


class TrainingWorld:
    def __init__(self, spec, training_seed, rank):
        self.spec, self.training_seed, self.rank = spec, training_seed, rank
        self.rng = random.Random((training_seed << 16) + rank)
        self.session = Session(spec)
        self.episode = 0
        self.reset()

    def reset(self):
        for _ in range(100):
            name = self.rng.choice(self.spec["maps"]["train"])
            seed = self.rng.randrange(100000, 2 ** 32)
            task = task_for(self.spec, name, seed, "train")
            task.task_id = "train_seed_{}__worker_{}__episode_{}".format(
                self.training_seed, self.rank, self.episode)
            self.episode += 1
            self.current = self.session.reset_task(task)
            if not self.session.terminated:
                break
        else:
            raise ValueError("100 training starts already achieve target; choose a harder map/range")
        self.ledger = EpisodeLedger(self.session, "mappo", self.training_seed)
        self.ledger.observe(self.session, MemoryRecorder())

    def step(self, request):
        final, reward, terminated, truncated, _ = self.session.step_request(request)
        sink = MemoryRecorder()
        self.ledger.observe(self.session, sink)
        if terminated or truncated:
            self.ledger.finish(self.session, sink)
            self.reset()
        else:
            self.current = final
        return {"observation": self.current, "final_observation": final,
                "reward": reward, "terminated": terminated, "truncated": truncated,
                "episodes": sink.episodes}


def _serve(connection, spec, seed, rank, restored):
    # The parent owns graceful checkpointing. A process-group Ctrl-C/TERM
    # must not kill a worker while the parent is finishing its rollout.
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, signal.SIG_IGN)
    try:
        world = restored if restored is not None else TrainingWorld(spec, seed, rank)
        connection.send((True, world.current))
        while True:
            command, argument = connection.recv()
            if command == "close":
                return
            if command == "step":
                value = world.step(argument)
            elif command == "snapshot":
                value = world
            else:
                raise ValueError("unknown world command")
            connection.send((True, value))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class WorldBatch:
    """One session per worker; state stays in that worker until checkpointed."""

    def __init__(self, spec, seed, restored=None):
        self.connections, self.processes = [], []
        count = spec["training"]["num_envs"]
        if restored is not None and len(restored) != count:
            raise ValueError("checkpoint worker count changed")
        self.local = None
        try:
            if count == 1:
                self.local = restored[0] if restored else TrainingWorld(spec, seed, 0)
                self.observations = [self.local.current]
            else:
                context = get_context("spawn")
                for rank in range(count):
                    parent, child = context.Pipe()
                    process = context.Process(target=_serve, args=(child, spec, seed, rank,
                                               restored[rank] if restored else None), daemon=True)
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                self.observations = [self._receive(pipe) for pipe in self.connections]
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection):
        if not connection.poll(300):
            raise RuntimeError("training worker did not respond in 300 seconds")
        okay, value = connection.recv()
        if not okay:
            raise RuntimeError("training worker failed:\n" + value)
        return value

    def step(self, requests):
        if self.local is not None:
            results = [self.local.step(requests[0])]
        else:
            for pipe, request in zip(self.connections, requests):
                pipe.send(("step", request))
            results = [self._receive(pipe) for pipe in self.connections]
        self.observations = [result["observation"] for result in results]
        return results

    def snapshot(self):
        if self.local is not None:
            return [self.local]
        for pipe in self.connections:
            pipe.send(("snapshot", None))
        return [self._receive(pipe) for pipe in self.connections]

    def close(self):
        for pipe in self.connections:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            pipe.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
