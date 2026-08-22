"""
음성 시스템 모듈

WebRTC + TTS/STT 연동:
- 텍스트 → 음성 (TTS)
- 음성 → 텍스트 (STT)
- 실시간 음성 통화 (WebRTC)
"""

import os
import json
from typing import Optional
from datetime import datetime


class VoiceSystem:
    def __init__(self):
        self.tts_provider = "browser"
        self.stt_provider = "browser"
        self.language = "ko-KR"

    def get_tts_config(self) -> dict:
        return {
            "provider": self.tts_provider,
            "language": self.language,
            "rate": 1.0,
            "pitch": 1.0,
            "volume": 1.0
        }

    def get_stt_config(self) -> dict:
        return {
            "provider": self.stt_provider,
            "language": self.language,
            "continuous": True,
            "interimResults": True
        }

    def get_webrtc_config(self) -> dict:
        return {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"}
            ],
            "iceCandidatePoolSize": 2
        }

    def get_status(self) -> dict:
        return {
            "tts_provider": self.tts_provider,
            "stt_provider": self.stt_provider,
            "language": self.language,
            "webrtc_supported": True
        }
