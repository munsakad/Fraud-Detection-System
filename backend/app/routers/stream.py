from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import simulator

router = APIRouter(tags=["stream"])


@router.websocket("/ws/live")
async def live_feed(websocket: WebSocket):
    await websocket.accept()
    simulator.register(websocket)
    try:
        # send a snapshot of recent events on connect so the UI isn't empty
        for event in reversed(simulator.recent_events(30)):
            await websocket.send_json(event)
        while True:
            # keep the connection open; we don't expect client messages, but
            # reading lets us detect disconnects promptly
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        simulator.unregister(websocket)
