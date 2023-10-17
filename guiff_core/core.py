import datetime
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional, Union

import ffmpeg

logger = logging.getLogger()


def trim_video(source, start_time, end_time, output_dir, output_filename):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, output_filename)
    start_datetime = datetime.datetime.strptime(start_time, "%M:%S.%f")
    end_datetime = datetime.datetime.strptime(end_time, "%M:%S.%f")

    start_seconds = (
        (start_datetime.minute) * 60
        + start_datetime.second
        + (start_datetime.microsecond * 0.000001)
    )
    duration = end_datetime - start_datetime
    duration = duration.seconds + (duration.microseconds * 0.000001)

    input_0: ffmpeg.nodes.FilterableStream = ffmpeg.input(source)
    if type(input_0) is ffmpeg.nodes.FilterableStream:
        output = input_0.output(
            output_path,
            ss=start_seconds,
            t=duration,
            vcodec="copy",
            acodec="copy",
        )
        print(output.compile())
        output.run()
    else:
        print(f"Unknown Type Error: {type(input_0)}")


def convert_to_webm(
    source: Union[str, List[str]],
    output_dir: str,
    options: Optional[Dict[str, Any]] = None,
):
    convert_container(source, ".webm", output_dir, options=options)


def convert_to_mp4(
    source: Union[str, List[str]],
    output_dir: str,
):
    convert_container(source, ".mp4", output_dir)


def convert_container(
    source: Union[str, List[str]],
    target_container: str,
    output_dir: str,
    options: Optional[Dict[str, Any]] = None,
):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    def convert(video_path: pathlib.Path):
        suffix = video_path.suffix
        video_name = video_path.stem

        if suffix in [".avi", ".mp4", ".webm"]:
            if suffix == target_container:
                logger.warning(
                    f"{video_path.name} already has container {target_container}"
                )

            input_0: ffmpeg.nodes.FilterableStream = ffmpeg.input(
                str(video_path.absolute())
            )
            target_path = os.path.join(output_dir, video_name) + target_container
            if options is not None:
                output = input_0.output(target_path, **options)
            else:
                output = input_0.output(target_path)

            print()
            logger.info("Run ->\r\n" + " ".join(output.compile()) + "\r\n\r\n")
            output.run()
        else:
            logger.warning(f"{video_path.name} file type not supported")

    if type(source) == str:
        source_path = pathlib.Path(source)
        if source_path.is_file():
            convert(source_path)
        else:
            for file_path in sorted(source_path.iterdir()):
                if file_path.is_file():
                    convert(file_path)
    else:
        for file_path in sorted(source):
            file_path = pathlib.Path(file_path)
            if file_path.is_file():
                convert(file_path)
