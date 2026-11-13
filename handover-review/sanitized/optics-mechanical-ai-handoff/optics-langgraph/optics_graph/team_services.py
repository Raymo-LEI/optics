"""Versioned file jobs for interchangeable pre-hardware services."""
from hashlib import sha256
import subprocess
import sys

from .settings import PROJECT, ROOT
from .sim_io import read_bytes, read_json

JOB_VERSION = "optics-prehardware-job/v1"
BLOB_LIMIT = 3 * 256_000 + 4096
STAGES = {"vision", "controller", "sim", "execution"}


def digest(raw):
    return sha256(raw).hexdigest()


def job_identity(job, raw):
    return {"version": JOB_VERSION, "job_id": job["job_id"], "stage": job["stage"],
            "request_sha256": digest(raw)}


class FileTeamServices:
    """Service commands are supplied by trusted installation code, never by run input."""

    def __init__(self, commands=None):
        commands = commands or {}
        if commands.keys() - STAGES:
            raise ValueError("Unknown pre-hardware service")
        self.commands = dict(commands)

    def invoke(self, stage, path, payload, *, files=(), timeout=40):
        from .team_artifacts import atomic, encode

        inputs = {name: digest(read_bytes(path / name, limit=BLOB_LIMIT)) for name in files}
        job = {"version": JOB_VERSION, "job_id": f"{path.parents[1].name}:{path.name}",
               "stage": stage, "payload": payload, "files": inputs}
        raw = encode(job)
        request = path / "service-request.json"
        if request.exists():
            if read_bytes(request) != raw:
                raise ValueError("Service attempt request changed")
        else:
            atomic(request, raw)
        answer = path / "service-response.json"
        if not answer.exists():
            python = PROJECT / ".venv-mujoco/bin/python" if stage in {"vision", "sim"} else sys.executable
            command = self.commands.get(stage, (str(python), "-E", "-s", "-m", "optics_graph.team_service_worker"))
            proc = subprocess.run([*command, str(path.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}, timeout=timeout)
            if proc.returncode != 0:
                raise ValueError(f"{stage} service exited with code {proc.returncode}")
        response = read_json(answer)
        identity = job_identity(job, raw)
        if (not isinstance(response, dict) or set(response) != {*identity, "result"}
                or any(response[key] != value for key, value in identity.items())
                or not isinstance(response["result"], dict)):
            raise ValueError("Service response does not match this request")
        if read_bytes(request) != raw:
            raise ValueError("Service changed the request file")
        if any(digest(read_bytes(path / name, limit=BLOB_LIMIT)) != sha for name, sha in inputs.items()):
            raise ValueError("Service changed an input artifact")
        return response["result"]
