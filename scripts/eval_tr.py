#!/usr/bin/env python3
"""Türkçe test sorularını çalışan bir llama-server'a (OpenAI uyumlu API) sorar,
cevapları ve hızı Markdown dosyasına yazar. Yalnızca Python standart kütüphanesi kullanır,
bu yüzden doğrudan Raspberry Pi üzerinde de çalışır.

Kullanım:
    python3 scripts/eval_tr.py --url http://localhost:8080 --out eval/sonuc.md
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

DEFAULT_SYSTEM = ("Senin adın RobotAI4B. Raspberry Pi 5 üzerinde çalışan, Türkçe konuşan yardımsever "
                  "bir robot asistansın. Kısa, net ve doğal Türkçe cevaplar verirsin. Emin olmadığın "
                  "konularda bunu açıkça söylersin ve bilgi uydurmazsın.")


def ask(url, system, prompt, max_tokens, temperature):
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Qwen3.5 gibi "düşünen" modellerde düşünmeyi kapat (Gemma 4'te de varsayılan kapalı)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    dt = time.time() - t0
    text = data["choices"][0]["message"]["content"]
    timings = data.get("timings", {})
    return text.strip(), dt, timings.get("predicted_per_second"), timings.get("prompt_per_second")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--prompts", default=str(Path(__file__).resolve().parent.parent / "eval/prompts_tr.jsonl"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--out", default="eval/sonuc.md")
    ap.add_argument("--title", default="RobotAI4B Türkçe değerlendirme")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts, encoding="utf-8") if l.strip()]
    lines = [f"# {args.title}\n"]
    speeds = []
    for i, p in enumerate(prompts, 1):
        text, dt, gen_tps, pp_tps = ask(args.url, args.system, p, args.max_tokens, args.temperature)
        if gen_tps:
            speeds.append(gen_tps)
        speed = f"{gen_tps:.1f} token/sn üretim" if gen_tps else f"{dt:.1f} sn"
        print(f"[{i}/{len(prompts)}] {p}\n{text}\n({speed})\n")
        lines += [f"## {i}. {p}\n", text + "\n", f"_{speed}, toplam {dt:.1f} sn_\n"]
    if speeds:
        avg = sum(speeds) / len(speeds)
        lines.insert(1, f"Ortalama üretim hızı: **{avg:.1f} token/sn**\n")
        print(f"Ortalama üretim hızı: {avg:.1f} token/sn")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Sonuçlar: {args.out}")


if __name__ == "__main__":
    main()
