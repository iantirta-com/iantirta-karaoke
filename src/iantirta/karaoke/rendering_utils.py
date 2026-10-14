

import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from iantirta.media._http._http import BaseHTTP

_logger = logging.getLogger(__name__)

class Renderer:
    _default_render_dir: Path = Path("~/.cache/iantirta/render").expanduser().resolve()

    def __init__(self):
        self.ensure_font_exists()

    @staticmethod
    def ensure_font_exists():
        # TODO: On windows need font manual installation.
        if os.name != "nt":
            # font_dir = Path("/usr/share/fonts/truetype") # os level
            font_dir = Path("~/.local/share/fonts").expanduser().resolve()
        else:
            font_dir = Path(os.environ.get('LOCALAPPDATA', 'C:')) / "Microsoft" / "Windows" / "Fonts"

        montserrat_dir = font_dir / "montserrat"

        if montserrat_dir.exists() and any(montserrat_dir.iterdir()):
            return True
        
        montserrat_dir.mkdir(parents=True, exist_ok=True)

        font_url = "https://github.com/JulietaUla/Montserrat/archive/refs/tags/v7.222.zip"
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmpfile:
            BaseHTTP().download(font_url, tmpfile)

            # Rewined the tempfile seek
            tmpfile.seek(0)

            with zipfile.ZipFile(tmpfile, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith('.ttf') and 'fonts/ttf/' in file_info.filename:
                        file_info.filename = Path(file_info.filename).name
                        zip_ref.extract(file_info, montserrat_dir)

        return True

    def render(
        self,
        video_path: str | Path,
        audio_paths: list[str | Path],
        duration: float,
        outpath: str | Path,
        *,
        ass_content: str | None = None,
        logger: logging.Logger | None = None
    ):
        try:
            video_path = Path(video_path)
            if outpath is None:
                safe_title = "".join(c for c in video_path.stem if c.isalnum() or c in ' -_').rstrip()
                outpath = video_path.with_name(safe_title + '.mkv')
            else:
                outpath = Path(outpath)
                safe_title = "".join(c for c in outpath.stem if c.isalnum() or c in ' -_').rstrip()

            render_cmds = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
            ]

            for path in audio_paths:
                render_cmds += ["-i", str(path)]
        
            if ass_content is not None:
                Renderer._default_render_dir.mkdir(parents=True, exist_ok=True)

                ass_path = Renderer._default_render_dir / f"{safe_title}.ass"
                ass_path.write_text(ass_content, encoding="utf-8-sig")

                scale_filter = "fps=30,scale=if(gt(iw/ih\\,16/9)\\,-1\\,1280):if(gt(iw/ih\\,16/9)\\,720\\,-1):flags=fast_bilinear,crop=1280:720"
                filters_list = [
                    "-filter_complex", f"[0:v]{scale_filter}[base];[base]ass=filename={ass_path.as_posix()}[subbed];[subbed]copy[outv]",
                    "-map", "[outv]",
                ]
                video_encoding = ["-c:v", "h264_nvenc", "-cq", "34", "-preset", "p1", "-tune", "hq", "-rc", "vbr", "-b:v", "0", "-maxrate", "4M", "-bufsize", "8M",]
            else:
                filters_list = ["-map", "0:v:0",]
                video_encoding = ["-c:v", "copy",]

            render_cmds += [
                *filters_list,
                "-map", "1:a:0",
                "-map", "0:a:0",
                "-metadata", f"title={safe_title} (Karaoke+ by iantirta Version)",
                "-metadata", "artist=iantirta-karaoke",
                "-metadata:s:a:0", "title=Karaoke", "-disposition:a:0", "default",
                "-metadata:s:a:1", "title=Original Song", "-disposition:a:1", "none",
                "-t", str(duration),
                *video_encoding,
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(outpath),
            ]

            (logger or _logger).info(f"Rendering video to: {outpath!s}")
            (logger or _logger).debug('With command: %s', " ".join(c for c in render_cmds))

            res = subprocess.run(render_cmds, check=True, capture_output=True)
    
            if res.returncode == 0:
                (logger or _logger).info("Successfully rendered!")
            else:
                (logger or _logger).error(f"⚠ FFmpeg failed to render video: {res.stderr.decode()}")

        except Exception as e:
            (logger or _logger).warning(f"⚠ Video Can't be rendered as to why: {e}")
            raise

        return outpath


if __name__ == "__main__":
    video_path = Path("../dataset/mahalini.mp4").expanduser().resolve()
    audio_paths = [video_path.with_suffix(".wav")]
    duration = 285
    outpath = video_path.with_suffix(".mkv")
    Renderer().render(video_path, audio_paths, duration, outpath)