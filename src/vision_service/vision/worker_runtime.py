import asyncio

from vision_service.vision.analysis import AnalyzedFrameResult, AnalyzedFrameStream


async def wait_for_worker_result(
    *,
    frame_stream: AnalyzedFrameStream,
    stop_event: asyncio.Event,
    after_token: int | None,
    require_detection: bool,
) -> AnalyzedFrameResult | None:
    result_task = asyncio.create_task(
        frame_stream.wait_for_result(
            after_token=after_token,
            require_detection=require_detection,
        ),
    )
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {result_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if stop_task in done:
        return None

    stop_task.cancel()
    await asyncio.gather(stop_task, return_exceptions=True)
    return await result_task
