#!/usr/bin/env python3
"""RobotAI4B ile terminalden Türkçe sohbet (llama-server'a bağlanır).
Yalnızca Python standart kütüphanesi kullanır; Pi'ye ek paket kurmanız gerekmez.

Kullanım:
    python3 pi/chat.py                       # varsayılan: http://127.0.0.1:8080
    python3 pi/chat.py --url http://robot.local:8080

Komutlar:  /yeni  -> sohbeti sıfırla,   /cik  -> çıkış
"""
import argparse
import json
import sys
import time
import urllib.request

SYSTEM_PROMPT = ("Senin adın RobotAI4B. Raspberry Pi 5 üzerinde çalışan, Türkçe konuşan yardımsever "
                 "bir robot asistansın. Kısa, net ve doğal Türkçe cevaplar verirsin. Emin olmadığın "
                 "konularda bunu açıkça söylersin ve bilgi uydurmazsın.")


def stream_chat(url, messages, temperature, max_tokens):
    body = {"messages": messages, "stream": True, "temperature": temperature,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {})
            if delta.get("content"):
                yield delta["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--history", type=int, default=8, help="Hatırlanacak en fazla soru-cevap sayısı")
    args = ap.parse_args()

    history = []
    print("RobotAI4B hazır. (/yeni: sıfırla, /cik: çıkış)\n")
    while True:
        try:
            user = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/cik", "/çık", "/exit"):
            break
        if user == "/yeni":
            history = []
            print("(Sohbet sıfırlandı)\n")
            continue

        history = history[-2 * args.history:]
        messages = [{"role": "system", "content": args.system}] + history + \
                   [{"role": "user", "content": user}]
        print("Robot: ", end="", flush=True)
        answer, n, t0 = [], 0, time.time()
        try:
            for piece in stream_chat(args.url, messages, args.temperature, args.max_tokens):
                print(piece, end="", flush=True)
                answer.append(piece)
                n += 1
        except OSError as e:
            print(f"\n[Sunucuya bağlanılamadı: {e}. pi/run_server.sh çalışıyor mu?]\n")
            continue
        dt = time.time() - t0
        print(f"\n\033[2m({n} parça, {dt:.1f} sn)\033[0m\n", file=sys.stdout)
        history += [{"role": "user", "content": user},
                    {"role": "assistant", "content": "".join(answer)}]


if __name__ == "__main__":
    main()
