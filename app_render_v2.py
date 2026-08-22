"""
심리상담 클리닉 - 가상 공간 프로덕션 버전

프로덕션 기능:
- Rate Limiting (slowapi)
- Security Headers (HSTS, CSP, X-Frame-Options 등)
- CORS Restricted Origins
- Health Check 엔드포인트
- 구조화된 JSON 로깅
- Graceful Shutdown
- HTTPS 지원 (CLI --ssl-keyfile/--ssl-certfile)
- ollama + qwen2.5:3b + 3계층 캐시
"""

import os
import sys
import json as json_module
import logging
import asyncio
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

# ── Structured JSON Logging ──────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "exc_info") and record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json_module.dumps(log_data, ensure_ascii=False)

_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from agents.evolution import AgentManager, AgentDNA, VirtualAgent
from agents.llm_client import LLMClient
from agents.cache import RealtimeCacheManager
from agents.ollama_pool import InferencePool, RequestPriority, get_pool, init_pool, shutdown_pool
from agents.redis_cache import redis_cache
from agents.redis_cache import redis_cache


async def autonomous_chat_loop():
    """30초마다 방별 에이전트끼리 자율 대화 실행"""
    while True:
        await asyncio.sleep(30)
        try:
            conversations = room_agent_manager.autonomous_chat()
            for conv in conversations:
                room = room_manager.get_room(conv["room_id"])
                if not room:
                    continue
                for ck, conn in list(room_manager.connections.items()):
                    if ck.startswith(conv["room_id"]):
                        try:
                            await conn.send_json({
                                "type": "autonomous_chat",
                                "room_id": conv["room_id"],
                                "agent1_id": conv["agent1_id"],
                                "agent1_name": conv["agent1_name"],
                                "message1": conv["message1"],
                                "agent2_id": conv["agent2_id"],
                                "agent2_name": conv["agent2_name"],
                                "message2": conv["message2"],
                                "timestamp": conv["timestamp"],
                            })
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"[AUTONOMOUS_CHAT] error: {e}")


ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:7862,http://localhost:7863,http://127.0.0.1:7862"
).split(",")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(autonomous_chat_loop())
    await init_pool()
    await redis_cache.connect()
    logger.info("Server starting up")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await shutdown_pool()
    await redis_cache.close()
    logger.info("Server shutting down gracefully")


app = FastAPI(title="심리상담 클리닉 - 가상空間", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    host = request.headers.get("host", "")
    if "localhost" not in host and "127.0.0.1" not in host:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

agent_manager = AgentManager()
llm_client = LLMClient()
realtime_cache = RealtimeCacheManager()


# ── 방별 에이전트 관리 ──────────────────────────────────────────────

class RoomAgentManager:
    """방별로 에이전트를 배정하고 자율 대화를 관리"""

    def __init__(self):
        self.room_agents: Dict[str, List[str]] = {}  # room_id -> [agent_id, ...]
        self._assign_initial_agents()

    def _assign_initial_agents(self):
        self.room_agents["lobby"] = ["observer-1", "counselor-2"]
        self.room_agents["counseling-1"] = ["counselor-1"]
        self.room_agents["counseling-2"] = []

    def get_agents_for_room(self, room_id: str) -> List[dict]:
        """방에 배정된 에이전트 목록 반환"""
        agent_ids = self.room_agents.get(room_id, [])
        result = []
        for aid in agent_ids:
            agent = agent_manager.get_agent(aid)
            if agent and agent.is_alive:
                result.append(agent.get_state())
        return result

    def move_agent(self, agent_id: str, from_room: str, to_room: str) -> bool:
        """에이전트를 방 간 이동"""
        if from_room in self.room_agents and agent_id in self.room_agents[from_room]:
            self.room_agents[from_room].remove(agent_id)
        if to_room not in self.room_agents:
            self.room_agents[to_room] = []
        self.room_agents[to_room].append(agent_id)
        logger.info(f"[AGENT] {agent_id} moved: {from_room} -> {to_room}")
        return True

    def autonomous_move(self):
        for room_id, agent_ids in list(self.room_agents.items()):
            for agent_id in agent_ids[:]:
                agent = agent_manager.get_agent(agent_id)
                if not agent or not agent.is_alive:
                    continue
                if agent.energy < 20:
                    continue

                # 특성 기반 이동 확률: curiosity(0.15) + social(0.1) + energy(0.05)
                move_prob = (
                    agent.dna.traits.get("curiosity", 0.5) * 0.15
                    + agent.dna.traits.get("social", 0.5) * 0.1
                    + agent.dna.traits.get("energy", 0.5) * 0.05
                )

                if random.random() < move_prob:
                    all_rooms = list(self.room_agents.keys())
                    if len(all_rooms) > 1:
                        other_rooms = [r for r in all_rooms if r != room_id]
                        target = random.choice(other_rooms)
                        self.move_agent(agent_id, room_id, target)

    def autonomous_chat(self) -> List[dict]:
        conversations = []
        topics = [
            "오늘 날씨가 좋네요", "상담이 중요한 것 같아요",
            "사람들의 마음은 복잡하죠", "함께 생각해보면 좋겠어요",
            "호기심이 많은 세상이에요", "논리적으로 접근하면 좋겠어요",
            "공감이 필요해 보여요", "새로운 관점이 필요하겠어요",
        ]

        for room_id, agent_ids in self.room_agents.items():
            if len(agent_ids) < 2:
                continue
            # 30% 확률로 대화 발생
            if random.random() > 0.3:
                continue

            a1_id, a2_id = random.sample(agent_ids, 2)
            a1 = agent_manager.get_agent(a1_id)
            a2 = agent_manager.get_agent(a2_id)
            if not a1 or not a2 or not a1.is_alive or not a2.is_alive:
                continue

            topic = random.choice(topics)
            response1_cached = realtime_cache.get(a1_id, topic)
            if response1_cached:
                response1 = response1_cached
            else:
                response1 = a1.respond(topic)
                realtime_cache.put(a1_id, topic, response1)

            prompt2 = f"{a1.name}: {response1}"
            response2_cached = realtime_cache.get(a2_id, prompt2)
            if response2_cached:
                response2 = response2_cached
            else:
                response2 = a2.respond(prompt2)
                realtime_cache.put(a2_id, prompt2, response2)

            conversations.append({
                "room_id": room_id,
                "agent1_id": a1_id, "agent1_name": a1.name, "message1": response1,
                "agent2_id": a2_id, "agent2_name": a2.name, "message2": response2,
                "timestamp": datetime.now().isoformat()
            })

        return conversations


room_agent_manager = RoomAgentManager()


class VirtualRoom:
    def __init__(self, room_id: str, name: str, max_users: int = 10):
        self.room_id = room_id
        self.name = name
        self.max_users = max_users
        self.users = {}
        self.messages = []

    def add_user(self, user_id, user_info) -> bool:
        if len(self.users) >= self.max_users:
            return False
        self.users[user_id] = {
            **user_info,
            "joined_at": datetime.now().isoformat(),
            "position": {"x": 100, "y": 100}
        }
        return True

    def remove_user(self, user_id):
        self.users.pop(user_id, None)

    def add_message(self, sender, message, msg_type="chat"):
        self.messages.append({
            "sender": sender, "message": message, "type": msg_type,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]

    def get_state(self):
        return {
            "room_id": self.room_id, "name": self.name,
            "users": self.users, "user_count": len(self.users),
            "max_users": self.max_users, "messages": self.messages[-20:]
        }


class RoomManager:
    def __init__(self):
        self.rooms = {}
        self.connections = {}
        self._create_defaults()

    def _create_defaults(self):
        self.create_room("lobby", "로비", 20)
        self.create_room("counseling-1", "상담방 1", 5)
        self.create_room("counseling-2", "상담방 2", 5)

    def create_room(self, room_id, name, max_users=10):
        if room_id not in self.rooms:
            self.rooms[room_id] = VirtualRoom(room_id, name, max_users)
        return self.rooms[room_id]

    def get_room(self, room_id):
        return self.rooms.get(room_id)

    def get_all_rooms(self):
        return [
            {"room_id": r.room_id, "name": r.name, "user_count": len(r.users), "max_users": r.max_users}
            for r in self.rooms.values()
        ]


room_manager = RoomManager()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/health")
async def health_check():
    pool = get_pool()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": llm_client.get_status()["provider"],
        "agents_alive": len([a for a in agent_manager.agents.values() if a.is_alive]),
        "rooms": len(room_manager.rooms),
        "pool": pool.stats,
    }


@app.get("/")
async def root():
    return RedirectResponse(url="/room/lobby")


@app.get("/room/{room_id}")
async def serve_room(room_id: str):
    room_path = os.path.join(BASE_DIR, "static", "room.html")
    if os.path.exists(room_path):
        with open(room_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>404</h1>", status_code=404)


@app.get("/api")
async def api_root():
    return {
        "message": "심리상담 클리닉 API",
        "version": "2.0.0",
        "endpoints": [
            "/api/rooms - 가상 방 목록",
            "/api/agents - 에이전트 목록",
            "/api/evolution - 진화 실행",
            "/api/stats - 통계",
            "/ws/{room_id}/{user_id} - WebSocket"
        ]
    }


@app.get("/api/rooms")
@limiter.limit("100/minute")
async def get_rooms(request: Request):
    return {"rooms": room_manager.get_all_rooms()}


@app.get("/api/rooms/{room_id}")
async def get_room(room_id: str):
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="방을 찾을 수 없습니다")
    return room.get_state()


@app.post("/api/rooms/{room_id}/join")
async def join_room(room_id: str, user_id: str, user_name: str):
    logger.info(f"[JOIN] room_id={room_id} user_id={user_id} user_name={user_name}")
    room = room_manager.get_room(room_id)
    if not room:
        logger.error(f"[JOIN] room not found: {room_id}")
        raise HTTPException(status_code=404, detail="방을 찾을 수 없습니다")
    success = room.add_user(user_id, {"name": user_name})
    if not success:
        logger.error(f"[JOIN] room full: {room_id}")
        raise HTTPException(status_code=400, detail="방이 가득 찼습니다")
    logger.info(f"[JOIN] success: {user_name} -> {room_id} (users: {len(room.users)})")
    return {"success": True, "room": room.get_state()}


@app.post("/api/rooms/{room_id}/leave")
async def leave_room(room_id: str, user_id: str):
    room = room_manager.get_room(room_id)
    if room:
        room.remove_user(user_id)
    return {"success": True}


@app.get("/api/room-agents/{room_id}")
async def get_room_agents(room_id: str):
    return {"agents": room_agent_manager.get_agents_for_room(room_id)}


@app.get("/api/agents")
@limiter.limit("100/minute")
async def get_agents(request: Request):
    return {"agents": agent_manager.get_all_agents()}


@app.post("/api/agents/create")
async def create_agent(name: str, prompt: str):
    agent_id = f"user-{name}-{datetime.now().strftime('%H%M%S')}"
    agent = agent_manager.create_agent(agent_id, name, prompt)
    return {"success": True, "agent": agent.get_state()}


@app.get("/api/evolution")
@limiter.limit("100/minute")
async def run_evolution(request: Request):
    record = agent_manager.run_evolution()
    return record


@app.get("/api/evolution/report")
async def evolution_report():
    return agent_manager.get_darwin_report()


@app.get("/api/cache/stats")
async def cache_stats():
    return {
        "realtime_cache": realtime_cache.stats,
        "llm_cache": llm_client.cache.stats,
    }


@app.get("/api/stats")
@limiter.limit("100/minute")
async def get_stats(request: Request):
    pool = get_pool()
    return {
        "rooms": len(room_manager.rooms),
        "total_agents": len(agent_manager.agents),
        "alive_agents": len([a for a in agent_manager.agents.values() if a.is_alive]),
        "llm": llm_client.get_status(),
        "evolution": agent_manager.get_stats(),
        "pool": pool.stats,
    }


@app.websocket("/ws/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    logger.info(f"[WS] connection request: room={room_id} user={user_id}")
    await websocket.accept()
    logger.info(f"[WS] accepted: room={room_id} user={user_id}")

    room = room_manager.get_room(room_id)
    if not room:
        logger.error(f"[WS] room not found: {room_id}")
        await websocket.close(code=4004, reason="방을 찾을 수 없습니다")
        return

    room_manager.connections[f"{room_id}:{user_id}"] = websocket
    room.add_message("system", f"{user_id}님이 입장했습니다", "system")

    state = room.get_state()
    logger.info(f"[WS] sending state to {user_id}: {len(state['users'])} users in room")
    await websocket.send_json({"type": "state", "room": state})

    for ck, conn in room_manager.connections.items():
        if ck.startswith(room_id) and ck != f"{room_id}:{user_id}":
            try:
                await conn.send_json({
                    "type": "user_join",
                    "user_id": user_id,
                    "user_info": room.users.get(user_id, {})
                })
            except Exception:
                pass

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "chat")

            if msg_type == "chat":
                message = data.get("message", "")
                room.add_message(user_id, message, "chat")
                for ck, conn in room_manager.connections.items():
                    if ck.startswith(room_id):
                        try:
                            await conn.send_json({
                                "type": "message", "user_id": user_id,
                                "message": message, "timestamp": datetime.now().isoformat()
                            })
                        except Exception:
                            pass

            elif msg_type == "agent_chat":
                agent_id = data.get("agent_id", "counselor-1")
                message = data.get("message", "")
                t0 = time.monotonic()
                cached = realtime_cache.get(agent_id, message)
                if cached:
                    response = cached
                else:
                    redis_hit = await redis_cache.get_inference(agent_id, message)
                    if redis_hit:
                        response = redis_hit
                        realtime_cache.put(agent_id, message, response)
                    else:
                        priority = RequestPriority.COUNSELOR if "counselor" in agent_id else RequestPriority.OBSERVER
                        pool = get_pool()
                        response = await pool.submit(
                            prompt=f"당신은 {agent_id}입니다. 사용자: {message}",
                            agent_id=agent_id,
                            priority=priority,
                            timeout=30.0,
                        )
                        realtime_cache.put(agent_id, message, response)
                        await redis_cache.set_inference(agent_id, message, response)
                latency_ms = (time.monotonic() - t0) * 1000
                room.add_message(agent_id, response, "agent")
                await websocket.send_json({
                    "type": "agent_response", "agent_id": agent_id,
                    "response": response, "latency_ms": round(latency_ms, 1),
                    "cached": cached is not None,
                })

            elif msg_type == "move":
                position = data.get("position", {"x": 0, "y": 0})
                if user_id in room.users:
                    room.users[user_id]["position"] = position
                for ck, conn in room_manager.connections.items():
                    if ck.startswith(room_id):
                        try:
                            await conn.send_json({
                                "type": "move", "user_id": user_id, "position": position
                            })
                        except Exception:
                            pass

            elif msg_type == "evolve":
                record = agent_manager.run_evolution()
                await websocket.send_json({"type": "evolution", "record": record})

    except WebSocketDisconnect:
        room_manager.connections.pop(f"{room_id}:{user_id}", None)
        room.remove_user(user_id)
        room.add_message("system", f"{user_id}님이 퇴장했습니다", "system")
        for ck, conn in room_manager.connections.items():
            if ck.startswith(room_id):
                try:
                    await conn.send_json({"type": "leave", "user_id": user_id})
                except Exception:
                    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    args = parser.parse_args()

    port = int(os.environ.get("PORT", args.port))
    ssl_kwargs = {}
    if args.ssl_keyfile and args.ssl_certfile:
        ssl_kwargs = {"ssl_keyfile": args.ssl_keyfile, "ssl_certfile": args.ssl_certfile}

    proto = "https" if ssl_kwargs else "http"
    print(f"=== 심리상담 클리닉 - 가상 공간 프로덕션 v2.0 ===")
    print(f"서버: {proto}://localhost:{port}")
    print(f"Health: {proto}://localhost:{port}/health")
    print(f"가상 공간: {proto}://localhost:{port}/room/lobby")
    print(f"API: {proto}://localhost:{port}/api")
    print(f"에이전트: {len(agent_manager.agents)}개")
    print(f"LLM: {llm_client.get_status()['provider']} / {llm_client.get_status()['model']}")
    print(f"CORS: {ALLOWED_ORIGINS}")
    if ssl_kwargs:
        print(f"HTTPS: enabled (keyfile={args.ssl_keyfile})")

    uvicorn.run(app, host=args.host, port=port, **ssl_kwargs)
