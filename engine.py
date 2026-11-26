import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath, PureWindowsPath

from filelock import FileLock

import vr180
from vr360_options import VIEW_ZOOM_RANGE, style_options

ROOT = Path(__file__).resolve().parent
STATE = Path(os.getenv("MMD_STATE", ROOT / ".runtime/state")).resolve()
INPUT = Path(os.getenv("MMD_INPUT", ROOT / "input"))
OUTPUT = Path(os.getenv("MMD_OUTPUT", ROOT / "output"))
INPUT_DISPLAY = os.getenv("MMD_INPUT_DISPLAY", str(INPUT))
OUTPUT_DISPLAY = os.getenv("MMD_OUTPUT_DISPLAY", str(OUTPUT))
NUNIF = Path(os.getenv("MMD_NUNIF", ROOT / ".runtime/nunif")).resolve()
MODELS = Path(os.getenv("NUNIF_HOME", ROOT / ".runtime/models")).resolve()
EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
PROFILES = {"轻立体": 1.0, "标准立体": 2.0, "重立体": 3.0}
PROFILE_NAMES = {value: name for name, value in PROFILES.items()}
MAX_DIVERGENCE = 5.0
CINEMA = "影院 Full SBS"
MODES = {CINEMA: 0, "VR180 75°": 75, "VR180 100°": 100, "VR180 110°": 110}
VR360_A = "ltx23-vr360-a1"
CONTEXT_STAGE = "vace-context-1"
REFERENCE_STAGE = "reference-only-1"
FIXED_STAGE = "klein-stage-3.3"
DANCE = "舞蹈类"
CLOSEUP = "H 类（人物近景）"
STAGE_METHODS = {
    "含角色扩画（VACE）": CONTEXT_STAGE,
    "固定舞台（FLUX）": FIXED_STAGE,
    "仅用空舞台（无匹配用环境光）": REFERENCE_STAGE,
}
WIDTHS = (0, 1280, 1920, 2560)  # 0 selects the automatic output size.
MODELS_ALLOWED = {"VDA_S", "VDA_Metric_L"}
VERSION = "q2-1.0-92c294e"
OUTPUT_NAME = re.compile(
    r"(?P<stem>.+)_(?:[ABC]|D\d+(?:\.\d+)?)(?:_MetricL)?_(?P<key>[0-9a-f]{8})"
    r"(?:_Zoom\d+(?:\.\d+)?)?(?:_Center)?(?:_Curve\d+)?"
    r"_(?:(?:H3|LTX25|VR360A|VACE|PlateOnly|Stage)_)?"
    r"(?:(?:FOV(?P<angle>75|100|110)_LR_180)|LRF_Full_SBS)\.mp4",
    re.IGNORECASE,
)
STATE.mkdir(exist_ok=True, parents=True)
(STATE / "logs").mkdir(exist_ok=True)


@contextlib.contextmanager
def db():
    conn = sqlite3.connect(STATE / "jobs.sqlite3", timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize():
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("BEGIN IMMEDIATE")
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT '',
            result TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', created REAL NOT NULL,
            updated REAL NOT NULL, cancel INTEGER NOT NULL DEFAULT 0)""")
        if "scan_existing" not in {column["name"] for column in c.execute("PRAGMA table_info(jobs)")}:
            c.execute("ALTER TABLE jobs ADD COLUMN scan_existing INTEGER NOT NULL DEFAULT 0")
        c.execute("""CREATE TABLE IF NOT EXISTS job_progress (
            job TEXT PRIMARY KEY, step INTEGER NOT NULL, frames INTEGER NOT NULL,
            log_offset INTEGER NOT NULL)""")


def media_path(value, root):
    """Translate mounted, host and configured share paths to a path under root."""
    base = root.resolve(strict=True)
    display = INPUT_DISPLAY if root == INPUT else OUTPUT_DISPLAY
    aliases = os.getenv(
        "MMD_INPUT_ALIASES" if root == INPUT else "MMD_OUTPUT_ALIASES", ""
    )
    raw = str(value).strip().strip('"')
    if not raw:
        raise ValueError("请填写路径")
    windows = PureWindowsPath(raw)
    if windows.drive or Path(raw).is_absolute():
        path_type = PureWindowsPath if windows.drive else PurePosixPath
        typed = path_type(raw)
        for alias in (str(root), str(base), display, *aliases.split("|")):
            if alias and typed.is_relative_to(path_type(alias)):
                relative = typed.relative_to(path_type(alias))
                break
        else:
            raise ValueError(f"路径须在 {display} 以内")
    else:
        relative = Path(raw)
    path = base.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"路径须在 {display} 以内")
    return path


def display_path(path, root):
    display = INPUT_DISPLAY if root == INPUT else OUTPUT_DISPLAY
    return str(Path(display) / path.relative_to(root.resolve()))


def source(rel):
    path = media_path(rel, INPUT)
    if path.suffix.lower() not in EXT or not path.is_file():
        raise ValueError("请选择素材目录内的视频")
    return path


def divergence(profile):
    """Read either a legacy preset name or a numeric iw3 strength."""
    if isinstance(profile, str) and profile in PROFILES:
        return PROFILES[profile]
    try:
        value = float(profile)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("立体强度应为 0.0–5.0，步长 0.1") from exc
    if (
        isinstance(profile, bool)
        or not 0 <= value <= MAX_DIVERGENCE
        or round(value, 1) != value
    ):
        raise ValueError("立体强度应为 0.0–5.0，步长 0.1")
    return value or 0.0


def empty_stages(angle, *, source_path=None):
    """A source binding selects the policy; geometry is checked again per shot."""
    stages = {}
    source_digest = None
    size = source_path.stat().st_size if source_path is not None else None
    for info in sorted((MODELS / "stage-library").glob("*/info.json")):
        data = json.loads(info.read_text(encoding="utf8"))
        if data.get("empty_stage") is not True or data.get("angle") != angle:
            continue
        if source_path is not None:
            if data.get("source_size") != size:
                continue
            if source_digest is None:
                with source_path.open("rb") as file:
                    source_digest = hashlib.file_digest(file, "sha256").hexdigest()
            if source_digest != data.get("source_sha256"):
                continue
        images = [info.parent / name for name in ("anchor.png", "panorama.png")]
        if all(image.is_file() for image in images):
            stages[info.parent.name] = [
                hashlib.sha256(info.read_bytes()).hexdigest(),
                *[p.stat().st_mtime_ns for p in images],
            ]
    return stages


def prepare_job(
    rel,
    start=0,
    length=0,
    profile="轻立体",
    width=0,
    model="VDA_S",
    pilot="",
    mode=CINEMA,
    curvature=0,
    keep_center=False,
    output_path=None,
    keep_folders=True,
    listing_root=".",
    auto_stage=False,
    stage_method=None,
    stage_preset=None,
    empty_only=False,
    fast_closeup=False,
    match_stage=True,
    seam_luma=True,
    generation_fps=24,
    seed=726391,
    observed_edges=True,
    anime_detail=True,
    view_zoom=1.0,
    **style,
):
    path = source(rel)
    start, length = float(start), float(length)
    if not all(math.isfinite(n) and n >= 0 for n in (start, length)):
        raise ValueError("起点与时长应为非负数")
    if width not in WIDTHS or mode not in MODES or model not in MODELS_ALLOWED:
        raise ValueError("无效预设")
    if isinstance(curvature, bool) or curvature not in range(101):
        raise ValueError("横向曲率须为 0–100 的整数百分比")
    if not isinstance(keep_center, bool):
        raise TypeError("保持中心人物大小须为布尔值")
    if not isinstance(auto_stage, bool):
        raise TypeError("自动舞台扩画须为布尔值")
    if stage_method is not None and stage_method not in STAGE_METHODS:
        raise ValueError("无效扩画方式")
    if stage_preset not in (None, DANCE, CLOSEUP) or not all(isinstance(v, bool) for v in (empty_only, fast_closeup)):
        raise ValueError("无效素材预设")
    if pilot and pilot not in [f"S{i:02}" for i in range(1, 7)]:
        raise ValueError("无效样片编号")
    strength = divergence(profile)
    stat = path.stat()
    spec = {
        "rel": path.relative_to(INPUT.resolve()).as_posix(),
        "start": start,
        "length": length,
        # Keep the same fingerprint as previously saved jobs at 1.0, 2.0 and 3.0.
        "profile": PROFILE_NAMES.get(strength, strength),
        "width": int(width),
        "model": model,
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "version": VERSION,
        "pilot": pilot,
    }
    default_folder = Path("Quest2试片") if pilot else Path(spec["rel"]).parent
    if output_path is not None or listing_root != ".":
        folder = media_path(OUTPUT if output_path is None else output_path, OUTPUT)
        if keep_folders:
            anchor = media_path(listing_root or ".", INPUT)
            # A directly pasted file can be outside the directory shown in the browser.
            subdir = (
                path.parent.relative_to(anchor)
                if not pilot and path.is_relative_to(anchor)
                else default_folder
            )
            folder = media_path(folder / subdir, OUTPUT)
        nearest = next(p for p in (folder, *folder.parents) if p.exists())
        if not nearest.is_dir() or not os.access(nearest, os.W_OK | os.X_OK):
            raise ValueError("输出路径需要是可写目录")
        relative = folder.relative_to(OUTPUT.resolve())
        if relative != default_folder:
            spec["output_dir"] = relative.as_posix()
    # Preserve existing cinema task fingerprints for identical manual settings.
    if mode != CINEMA:
        spec["mode"] = mode
        if type(view_zoom) not in (int, float) or not VIEW_ZOOM_RANGE[0] <= view_zoom <= VIEW_ZOOM_RANGE[1]:
            raise ValueError("正前方放大须在 1.00–1.25 倍内")
        if view_zoom != 1.0:
            spec["view_zoom"] = float(view_zoom)
        if auto_stage:
            switches = (match_stage, seam_luma, observed_edges, anime_detail)
            if not all(isinstance(value, bool) for value in switches):
                raise ValueError("VR360 A 开关必须为布尔值")
            if (generation_fps not in (12, 16, 24) or isinstance(seed, bool)
                    or not isinstance(seed, (int, float)) or not 0 <= seed < 2**31 or int(seed) != seed):
                raise ValueError("生成帧率须为 12、16 或 24，种子须为 0–2147483647 的整数")
            spec["stage_version"] = VR360_A
            spec["vr360"] = dict(
                match_stage=match_stage, seam_luma=seam_luma, generation_fps=int(generation_fps),
                seed=int(seed), observed_edges=observed_edges, anime_detail=anime_detail,
                **style_options(style),
            )
        if curvature:
            spec["curvature"] = int(curvature)
            if keep_center:
                spec["keep_center"] = True
    packed = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(packed.encode()).hexdigest()[:16]
    return key, packed


def insert_jobs(jobs, scan_existing=None):
    queued = 0
    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        for key, packed in jobs:
            now = time.time()
            scan = scan_existing.get(key, 0) if scan_existing else 0
            inserted = c.execute(
                """INSERT OR IGNORE INTO jobs(id,spec,status,created,updated,scan_existing)
                   VALUES(?,?,'queued',?,?,?)""",
                (key, packed, now, now, scan),
            ).rowcount
            if inserted:
                queued += 1
                continue
            row = c.execute("SELECT status,result FROM jobs WHERE id=?", (key,)).fetchone()
            if row["status"] == "done" and not Path(row["result"]).is_file():
                c.execute(
                    "UPDATE jobs SET status='queued', stage='', result='', updated=?,scan_existing=? WHERE id=?",
                    (now, scan, key),
                )
                queued += 1
    return [key for key, _ in jobs], queued


def enqueue_many(paths, *args, **options):
    # Finish filesystem and model checks before taking the SQLite write lock.
    jobs = [prepare_job(path, *args, **options) for path in paths]
    return insert_jobs(jobs)


def output_identity(spec):
    return job_output_folder(spec), Path(spec["rel"]).stem[:65].casefold()


def indexed_outputs(folder):
    found = {}
    try:
        for file in folder.iterdir():
            match = OUTPUT_NAME.fullmatch(file.name)
            if match and not file.is_symlink() and file.is_file():
                name = match["stem"].casefold()
                angle = int(match["angle"] or 0)
                found.setdefault((name, angle), []).append((file, match["key"].lower()))
    except FileNotFoundError:
        pass
    return found


def completed_output(spec, index=None, require_known_source=False):
    """Find a full, playable delivery for this source and requested projection."""
    if spec.get("pilot") or spec.get("start") or spec.get("length"):
        return None
    folder, name = output_identity(spec)
    angle = MODES[spec.get("mode", CINEMA)]
    if index is None:
        index = indexed_outputs(folder)
    matches = index.get((name, angle), ())
    if not matches:
        return None
    try:
        source_meta = probe(source(spec["rel"]))
        duration = float(source_meta["format"]["duration"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None
    source_has_audio = any(stream["codec_type"] == "audio" for stream in source_meta["streams"])
    custom = spec.get("vr360", {}).get("custom_model")
    for file, prefix in matches:
        with db() as connection:
            records = connection.execute("SELECT spec FROM jobs WHERE id LIKE ?", (prefix + "%",)).fetchall()
        related = [json.loads(row["spec"]) for row in records]
        if related and not any(
            old["rel"] == spec["rel"] and not old.get("pilot") and not old.get("start") and not old.get("length")
            and (old.get("size"), old.get("mtime")) == (spec.get("size"), spec.get("mtime"))
            for old in related
        ):
            continue
        if custom and not any(old == spec for old in related):
            continue
        if require_known_source and not related:
            continue
        try:
            stat = file.stat()
            if stat.st_size == 0 or stat.st_mtime_ns < spec["mtime"]:
                continue
            meta = probe(file)
            stream = video(meta)
            if stream["codec_name"] != "hevc":
                continue
            if source_has_audio and not any(item["codec_type"] == "audio" for item in meta["streams"]):
                continue
            if angle:
                side = {entry["side_data_type"] for entry in stream.get("side_data_list", ())}
                if stream["width"] != 2 * stream["height"] or not {"Stereo 3D", "Spherical Mapping"} <= side:
                    continue
            elif stream["width"] <= 2 * stream["height"]:
                continue
            span = float(stream.get("duration") or meta["format"]["duration"])
            if math.isfinite(span) and abs(span - duration) <= 0.35:
                return file
        except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError):
            continue
    return None


def enqueue_batch(paths, **options):
    jobs = [prepare_job(path, **options) for path in paths]
    specs = [json.loads(packed) for _, packed in jobs]
    identities = [output_identity(spec) for spec in specs]
    identities_count = {}
    for identity in identities:
        identities_count[identity] = identities_count.get(identity, 0) + 1
    indexes = {}
    pending = []
    scan = {}
    skipped = 0
    for job, spec, identity in zip(jobs, specs, identities):
        folder, _ = identity
        if folder not in indexes:
            indexes[folder] = indexed_outputs(folder)
        if completed_output(spec, indexes[folder], require_known_source=identities_count[identity] > 1):
            skipped += 1
        else:
            pending.append(job)
            scan[job[0]] = 2 if identities_count[identity] > 1 else 1
    keys, queued = insert_jobs(pending, scan)
    return keys, queued, skipped


def enqueue(rel, *args, **options):
    keys, _ = enqueue_many([rel], *args, **options)
    return keys[0]


def rows():
    with db() as c:
        return [
            dict(r)
            for r in c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 200")
        ]


def control(key, action):
    with db() as c:
        if action == "cancel":
            n = c.execute(
                """UPDATE jobs SET cancel=1, updated=?, status=CASE WHEN status='queued'
                          THEN 'cancelled' ELSE status END WHERE id=? AND status IN ('queued','running')""",
                (time.time(), key),
            ).rowcount
        elif action == "retry":
            n = c.execute(
                """UPDATE jobs SET status='queued', stage='', error='', cancel=0, updated=?
                          WHERE id=? AND status IN ('failed','interrupted','cancelled')""",
                (time.time(), key),
            ).rowcount
        else:
            raise ValueError(action)
    return "已更新" if n else "当前状态无需此操作"


def stop_queue():
    with db() as c:
        c.execute("BEGIN IMMEDIATE")
        queued = c.execute(
            "UPDATE jobs SET status='cancelled',stage='排队时取消',cancel=1,updated=? WHERE status='queued'",
            (time.time(),),
        ).rowcount
        running = c.execute(
            "UPDATE jobs SET cancel=1,updated=? WHERE status='running'", (time.time(),)
        ).rowcount
    if running:
        return f"正在停止转制并释放 GPU，另已取消 {queued} 个排队任务。"
    return f"已取消 {queued} 个排队任务。当前没有运行中的转制进程。"


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        encoding="utf8",
        timeout=120,
    )
    return json.loads(result.stdout)


def video(meta):
    return next(
        s
        for s in meta["streams"]
        if s["codec_type"] == "video"
        and not s.get("disposition", {}).get("attached_pic")
    )


def stream_interval(path, stream):
    """Return presentation bounds, including containers without stream durations."""
    start = stream.get("start_time")
    duration = stream.get("duration")
    if start not in (None, "N/A") and duration not in (None, "N/A"):
        return float(start), float(start) + float(duration)
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", str(stream["index"]),
         "-show_packets", "-show_entries", "packet=pts_time,duration_time",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, encoding="utf8",
    )
    first, last = math.inf, -math.inf
    for line in result.stdout.splitlines():
        fields = line.split(",")
        try:
            pts = float(fields[0])
        except ValueError:
            continue
        span = float(fields[1]) if len(fields) > 1 and fields[1] not in ("", "N/A") else 0
        first, last = min(first, pts), max(last, pts + span)
    return (first, last) if first <= last else (0.0, 0.0)


def validate_audio(source_path, source_meta, output_path, output_meta, start, length):
    original = next((s for s in source_meta["streams"] if s["codec_type"] == "audio"), None)
    encoded = next((s for s in output_meta["streams"] if s["codec_type"] == "audio"), None)
    if original is None:
        if encoded is not None:
            raise ValueError("无声源片出现意外音轨")
        return
    cut = float(source_meta["format"].get("start_time") or 0) + start
    first, last = stream_interval(source_path, original)
    expected_start = max(0.0, first - cut)
    expected_end = min(length, last - cut)
    if expected_end <= expected_start:
        if encoded and stream_interval(output_path, encoded)[1] > 0.16:
            raise ValueError("无声区间出现意外音轨")
        return
    if not encoded:
        raise ValueError("缺少原片音轨")
    actual_start, actual_end = stream_interval(output_path, encoded)
    if max(abs(actual_start - expected_start), abs(actual_end - expected_end)) > 0.16:
        raise ValueError(
            f"音轨区间校验失败，预期 {expected_start:.3f}–{expected_end:.3f}s，"
            f"实际 {actual_start:.3f}–{actual_end:.3f}s"
        )


def fps(stream):
    for name in ("avg_frame_rate", "r_frame_rate"):
        try:
            n = Fraction(stream[name])
            if n > 0:
                return n
        except (KeyError, ValueError, ZeroDivisionError):
            continue
    raise ValueError("无法确定源帧率")


def canvas(stream, cap):
    w, h = stream["width"], stream["height"]
    sar = stream.get("sample_aspect_ratio", "1:1")
    if sar not in (None, "N/A", "0:1"):
        w *= float(Fraction(sar.replace(":", "/")))
    for item in stream.get("side_data_list", []):
        if abs(item.get("rotation", 0)) % 180 == 90:
            w, h = h, w
    if cap == 0:
        max_w, max_h = (2560, 1440) if w >= h else (1440, 2560)
    else:
        max_w, max_h = cap, cap * 9 / 16
    scale = min(1, max_w / w, max_h / h)
    return max(2, 2 * round(w * scale / 2)), max(2, 2 * round(h * scale / 2))


@dataclass(frozen=True)
class OutputSize:
    source_eye_width: int
    source_eye_height: int
    eye_width: int
    eye_height: int

    @property
    def packed(self):
        return self.eye_width * 2, self.eye_height


def output_size(stream, width, mode):
    if width not in WIDTHS or mode not in MODES:
        raise ValueError("无效的分辨率或播放模式")
    w, h = canvas(stream, width)
    angle = MODES[mode]
    if not angle:
        return OutputSize(w, h, w, h)
    vertical = vr180.vertical_fov(w, h, angle)
    target = max(w * 180 / angle, h * 180 / vertical)
    edge = min(width or 2880, 2 * math.ceil(target / 2))
    return OutputSize(w, h, edge, edge)


class Cancelled(Exception):
    pass


def check_cancel(key):
    with db() as c:
        if c.execute("SELECT cancel FROM jobs WHERE id=?", (key,)).fetchone()[0]:
            raise Cancelled()


def set_stage(key, title, step, frames=0):
    check_cancel(key)
    log_path = STATE / "logs" / f"{key}.log"
    with log_path.open("ab") as log:
        log.write(("\n" + title + "\n").encode())
        offset = log.tell()
    with db() as c:
        c.execute(
            "UPDATE jobs SET stage=?,updated=? WHERE id=?", (title, time.time(), key)
        )
        c.execute(
            "INSERT OR REPLACE INTO job_progress VALUES (?,?,?,?)",
            (key, step, frames, offset),
        )
    return log_path


def run_step(key, title, cmd, step, frames, env=None, cwd=None):
    log_path = set_stage(key, title, step, frames)
    with log_path.open("ab") as log:
        started = time.monotonic()
        proc = subprocess.Popen(
            [str(x) for x in cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=cwd,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            start_new_session=os.name == "posix",
        )
        try:
            while proc.poll() is None:
                check_cancel(key)
                time.sleep(0.6)
        finally:
            if proc.poll() is None:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                else:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if proc.poll() is None:
                        proc.kill()
                proc.wait()
        if proc.returncode:
            raise RuntimeError(f"{title}失败，退出码 {proc.returncode}。详见任务日志。")
        log.write(f"\n{title}耗时 {time.monotonic() - started:.1f} 秒\n".encode())


def job_output_folder(spec):
    default = Path("Quest2试片") if spec["pilot"] else Path(spec["rel"]).parent
    return media_path(spec.get("output_dir", default), OUTPUT)


def transcode(key, s):
    path = source(s["rel"])
    if (path.stat().st_size, path.stat().st_mtime_ns) != (s["size"], s["mtime"]):
        raise ValueError("源文件已变化，请重新提交")
    meta = probe(path)
    v = video(meta)
    if v.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ValueError("此版本仅支持 SDR；检测到 HDR 源片")
    stereo_name = re.search(
        r"(?:^|[ _.-])(?:full[ _-]?sbs|hsbs|sbs|vr180|vr360)(?:$|[ _.-])",
        path.stem,
        re.IGNORECASE,
    )
    if stereo_name or any(
        x.get("side_data_type") in ("Stereo 3D", "Spherical Mapping")
        for x in v.get("side_data_list", [])
    ):
        raise ValueError("文件已有立体或全景标记，请检查原片")
    duration = float(meta["format"]["duration"])
    length = min(s["length"] or duration, duration - s["start"])
    if length < 0.25:
        raise ValueError("片段在文件末尾之外或不足 0.25 秒")
    rate = min(fps(v), Fraction(60))
    for standard in [
        Fraction(24000, 1001),
        Fraction(24),
        Fraction(25),
        Fraction(30000, 1001),
        Fraction(30),
        Fraction(50),
        Fraction(60000, 1001),
        Fraction(60),
    ]:
        if abs(rate - standard) < 0.002:
            rate = standard
            break
    mode = s.get("mode", CINEMA)
    angle = MODES[mode]
    curvature = s.get("curvature", 0) if angle else 0
    view_zoom = s.get("view_zoom", 1.0)
    size = output_size(v, s["width"], mode)
    w, h = size.source_eye_width, size.source_eye_height
    ten = any(bit in v.get("pix_fmt", "") for bit in ("10", "12", "16"))
    pix = "yuv420p10le" if ten else "yuv420p"
    name = s["pilot"] or path.stem[:65]
    strength = divergence(s["profile"])
    level = {1.0: "A", 2.0: "B", 3.0: "C"}.get(strength, f"D{strength:.1f}")
    if s["model"] == "VDA_Metric_L":
        level += "_MetricL"
    folder = job_output_folder(s)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = f"FOV{angle}_LR_180" if angle else "LRF_Full_SBS"
    if s.get("stage_version"):
        backend = s.get("vr360", {}).get("outpaint_model")
        vr360_prefix = {"h3": "H3_", "ltx25": "LTX25_"}.get(backend, "VR360A_")
        prefix = {VR360_A: vr360_prefix, CONTEXT_STAGE: "VACE_", REFERENCE_STAGE: "PlateOnly_"}.get(s["stage_version"], "Stage_")
        suffix = prefix + suffix
    if curvature:
        suffix = f"Curve{curvature}_{suffix}"
        if s.get("keep_center"):
            suffix = f"Center_{suffix}"
    if view_zoom != 1.0:
        suffix = f"Zoom{view_zoom:g}_{suffix}"
    dest = folder / f"{name}_{level}_{key[:8]}_{suffix}.mp4"
    common = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-stats_period",
        "1",
        "-y",
    ]
    frames = max(1, round(length * rate))
    env = dict(
        os.environ,
        NUNIF_HOME=str(MODELS),
        PYTHONUTF8="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(NUNIF),
        TORCH_HOME=str(MODELS / "torch"),
    )
    with tempfile.TemporaryDirectory(dir=STATE, prefix=f"work-{key}-") as tmp:
        work = Path(tmp)
        clip = work / "clip.mkv"
        stereo = work / "stereo.mp4"
        # Exact cut and one lossless normalized master keep model input and audio aligned.
        vf = f"fps={rate},scale={w}:{h}:flags=lanczos,setsar=1,format={pix}"
        cmd = common + [
            "-ss",
            s["start"],
            "-i",
            path,
            "-t",
            length,
            "-map",
            f"0:{v['index']}",
            "-map",
            "0:a:0?",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-threads",
            "6",
            "-preset",
            "ultrafast",
            "-qp",
            "0",
            "-c:a",
            "pcm_s24le",
            clip,
        ]
        run_step(key, "准备片段", cmd, 1, frames)
        cmd = [
            sys.executable,
            "-m",
            "iw3",
            "-i",
            clip,
            "-o",
            stereo,
            "--depth-model",
            s["model"],
            "--method",
            "row_flow_v3",
            "--divergence",
            strength,
            "--convergence",
            "0.5",
            "--scene-detect",
            "--ema-normalize",
            "--max-fps",
            "120",
            "--resolution",
            "392",
            "--batch-size",
            str(s.get("vr360", {}).get("iw3_batch", 1)),
            "--video-codec",
            "hevc_nvenc",
            "--preset",
            "p5",
            "--crf",
            "19",
            "--pix-fmt",
            pix,
            "-y",
        ]
        run_step(
            key,
            f"估深与双眼合成（立体强度 {strength:.1f}）",
            cmd,
            2,
            frames,
            env,
            NUNIF,
        )
        final = work / "final.mp4"
        stage_path = None
        if angle and s.get("stage_version"):
            context_stage = s["stage_version"] == CONTEXT_STAGE
            baseline = s["stage_version"] == VR360_A
            run_step(
                key,
                "H3 FL2VA 连续原帧扩画" if s.get("vr360", {}).get("outpaint_model") == "h3" else "VR360 A 连续原帧扩画" if baseline else "含角色视频扩画与边缘融合" if context_stage else "自动舞台扩画与跟踪",
                [
                    sys.executable,
                    "-u",
                    ROOT / ("stage_vr360.py" if baseline else "stage_context.py" if context_stage else "stage_background.py"),
                    work,
                    key,
                    size.eye_width,
                    angle,
                    rate,
                ],
                3,
                frames,
                env,
            )
            stage_path = work / "stage.mkv"
        projection_step = 4 if stage_path else 3
        set_stage(key, "准备投影和封装", projection_step)
        if angle:
            encoded = video(probe(stereo))
            video_args = vr180.projection_args(
                w,
                h,
                angle,
                size.eye_width,
                rate,
                ten,
                curvature=curvature,
                keep_center=s.get("keep_center", False),
                view_zoom=view_zoom,
                work=work,
                stage_path=stage_path,
                stereo_stage=s.get("stage_version") == VR360_A,
                detail_path=work / "vr360-detail.mkv" if s.get("vr360", {}).get("anime_detail") else None,
                colorspace=encoded.get("color_space"),
            )
            for param, field in [
                ("-colorspace", "color_space"),
                ("-color_trc", "color_transfer"),
                ("-color_primaries", "color_primaries"),
                ("-color_range", "color_range"),
            ]:
                if encoded.get(field) not in (None, "unknown", "unspecified"):
                    video_args.extend([param, encoded[field]])
        else:
            video_args = ["-map", "0:v:0", "-c:v", "copy", "-movflags", "+faststart"]
        cmd = common + [
            "-i",
            stereo,
            "-i",
            clip,
            *video_args,
            "-map",
            "1:a:0?",
            "-tag:v",
            "hvc1",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            final,
        ]
        title = "VR180 投影与封装" if angle else "封装音轨"
        opts = s.get("vr360", {})
        if stage_path and opts.get("contour_enabled") and opts["contour_strength"] > 0:
            cmd = vr180.contour_command(
                cmd, work, stereo, clip, w, h, angle, size.eye_width, rate, ten,
                curvature, encoded, opts["contour_strength"], opts["contour_width"],
                view_zoom=view_zoom,
            )
            title = "VR180 投影、二维外围接缝融合与封装"
        run_step(
            key,
            title,
            cmd,
            projection_step,
            frames,
        )
        if angle:
            vr180.add_projection_metadata(final)
        result = probe(final)
        stream = video(result)
        if (
            stream["codec_name"] != "hevc"
            or (stream["width"], stream["height"]) != size.packed
        ):
            raise ValueError("编码或双眼尺寸校验失败")
        if angle:
            sides = stream.get("side_data_list", [])
            if not (
                any(
                    x.get("type") == "side by side" and not x.get("inverted")
                    for x in sides
                )
                and any(x.get("projection") == "tiled equirectangular" for x in sides)
            ):
                raise ValueError("VR180 投影或左右眼标记校验失败")
        if abs(float(fps(stream) - rate)) > 0.02 or abs(
            float(stream["duration"]) - length
        ) > max(0.13, 2 / float(rate)):
            raise ValueError("时长或帧率校验失败")
        validate_audio(path, meta, final, result, s["start"], length)
        run_step(
            key,
            "校验视频解码",
            common + ["-xerror", "-i", final, "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"],
            projection_step + 1,
            frames,
        )
        set_stage(key, "发布成片", projection_step + 2)
        with tempfile.NamedTemporaryFile(
            dir=folder, prefix=f".{key}-", suffix=".part", delete=False
        ) as file:
            partial = Path(file.name)
        try:
            shutil.copyfile(final, partial)
            with db() as c:
                c.execute("BEGIN IMMEDIATE")
                if c.execute("SELECT cancel FROM jobs WHERE id=?", (key,)).fetchone()[
                    0
                ]:
                    raise Cancelled()
                os.replace(partial, dest)
                c.execute(
                    "UPDATE jobs SET status='done',stage='已验证',result=?,updated=? WHERE id=?",
                    (str(dest), time.time(), key),
                )
        finally:
            partial.unlink(missing_ok=True)


def worker():
    with FileLock(STATE / "worker.lock", timeout=0):
        with db() as c:
            c.execute(
                "UPDATE jobs SET status='interrupted',stage='上次运行中断' WHERE status='running'"
            )
        while True:
            with db() as c:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
                ).fetchone()
                if row:
                    c.execute("DELETE FROM job_progress WHERE job=?", (row["id"],))
                    c.execute(
                        "UPDATE jobs SET status='running',stage='探测素材',updated=? WHERE id=?",
                        (time.time(), row["id"]),
                    )
            if not row:
                time.sleep(1)
                continue
            try:
                spec = json.loads(row["spec"])
                scan = row["scan_existing"]
                existing = completed_output(spec, require_known_source=scan == 2) if scan else None
                if existing is not None:
                    with db() as c:
                        saved = c.execute(
                            """UPDATE jobs SET status='done',stage='已有成片，跳过转制',result=?,updated=?
                               WHERE id=? AND status='running' AND cancel=0""",
                            (str(existing), time.time(), row["id"]),
                        ).rowcount
                    if not saved:
                        raise Cancelled()
                else:
                    transcode(row["id"], spec)
            except Exception as e:  # noqa: BLE001 -- record job failures and keep the worker available
                status = "cancelled" if isinstance(e, Cancelled) else "failed"
                with db() as c:
                    c.execute(
                        "UPDATE jobs SET status=?,error=?,updated=? WHERE id=?",
                        (status, str(e), time.time(), row["id"]),
                    )


if __name__ == "__main__":
    initialize()
    worker()
