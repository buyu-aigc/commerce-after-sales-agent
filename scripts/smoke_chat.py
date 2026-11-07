from __future__ import annotations

import uuid

import httpx


def main() -> None:
    session_id = f"smoke-{uuid.uuid4().hex[:8]}"
    messages = [
        "health beauty 类目有什么运营风险？",
        "帮我写一个操作系统内核",
        "查订单 203096f03d82e0dffbc41ebc2e2bcfb7 状态，并且说明退款政策，然后生成售后升级话术",
        "no",
        "退款补偿能不能直接承诺？",
    ]
    print(f"session={session_id}")
    with httpx.Client(timeout=30) as client:
        for message in messages:
            response = client.post(
                "http://127.0.0.1:8001/chat",
                json={"message": message, "session_id": session_id},
            )
            response.raise_for_status()
            answer = response.json()["answer"].replace("\n", " | ")
            print(f"USER: {message}")
            print(f"AGENT: {answer[:700]}")
            print()


if __name__ == "__main__":
    main()
