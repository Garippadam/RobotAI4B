# RobotAI4B

Raspberry Pi 5 (16 GB) üzerinde **internetsiz** çalışan, Türkçesi iyi, ~4 milyar parametrelik yapay zekâ.

- **Temel model:** [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (Google, Apache 2.0 lisansı).
  "E4B" = her token'da ~4B *etkin* parametre çalışır. Türkçe dahil çok dilli eğitilmiş.
- **İnce ayar:** LoRA / QLoRA ile Türkçe soru-cevap verisi ve robot kişiliği (`data/robotai_persona.jsonl`).
- **Pi'de çalıştırma:** [llama.cpp](https://github.com/ggml-org/llama.cpp), 4-bit GGUF (~5 GB), OpenAI uyumlu API.

## Neden Gemma 4 E4B?

Kasım 2026 itibarıyla ~4B sınıfındaki iki güçlü, açık lisanslı aday aynı 12 Türkçe soruyla
(`eval/prompts_tr.jsonl`) karşılaştırıldı. İkisi de Q4_K_M, 4 CPU çekirdeğinde llama.cpp ile:

| | Gemma 4 E4B-it | Qwen3.5-4B |
|---|---|---|
| Türkiye'nin coğrafi bölgeleri | 7 bölge, doğru | "8 bölge", listeye "Trakya" eklemiş ✗ |
| Yazım düzeltme ("söylemiyo") | Tamamını düzeltti | "söylemiyo"yu bırakmış ✗ |
| Mercimek çorbası | Doğru tarif | Yumurta, "1 gece bekletin" ✗ |
| Atasözü, matematik, kod | Doğru | Doğru |
| GGUF boyutu (Q4_K_M) | 4.6 GiB | 2.5 GiB |
| Üretim hızı (aynı makine, `llama-bench`) | 8.2 token/sn | 6.8 token/sn |

Gemma 4 E4B dosya olarak daha büyük (katman-başı gömme tabloları yüzünden ~7.5B toplam parametre)
ama her token'da yalnızca ~4B parametre hesaplandığı için daha yavaş değil; 16 GB'lık Pi'ye de
rahatça sığıyor. Cevapların tamamı: `eval/baseline_gemma4_e4b.md`, `eval/karsilastirma_qwen3.5_4b.md`.

Model değiştirmek isterseniz (ör. daha hızlı ama daha zayıf `gemma-4-E2B-it`) sadece
`configs/robotai4b.yaml` içindeki `base_model` ve `pi/robotai4b.env` içindeki model dosyasını değiştirin.

## Klasör yapısı

```
configs/robotai4b.yaml        Model, veri kaynakları, LoRA ve eğitim ayarları
data/robotai_persona.jsonl    Robot kişiliği örnekleri (kendi örneklerinizi buraya ekleyin)
scripts/prepare_data.py       Veri kaynaklarını indirir, temizler, train/val dosyalarını yazar
scripts/train_lora.py         LoRA / QLoRA eğitimi (sadece asistan cevapları üzerinden loss)
scripts/merge_lora.py         Adaptörü temel modelle birleştirir
scripts/export_gguf.sh        Birleştirilmiş modeli GGUF'a çevirir + nicemler (Q4_K_M, Q4_0 ...)
scripts/export_lora_gguf.sh   Sadece LoRA adaptörünü GGUF'a çevirir (birleştirme gerekmez)
scripts/eval_tr.py            Türkçe test soruları + hız ölçümü (Pi'de de çalışır)
eval/prompts_tr.jsonl         Test soruları
notebooks/RobotAI4B_Colab.ipynb  Colab'da tek tıkla eğitim
pi/                           Raspberry Pi kurulum, sunucu, sohbet ve systemd servisi
```

## 1) Hızlı başlangıç: Pi'de hemen çalıştır (eğitimsiz)

64-bit Raspberry Pi OS üzerinde:

```bash
git clone https://github.com/garippadam/RobotAI4B && cd RobotAI4B
bash pi/setup_pi.sh          # llama.cpp'yi derler (~10 dk), modeli indirir (~5 GB)
bash pi/run_server.sh        # sunucu: http://127.0.0.1:8080  (tarayıcı arayüzü de var)
python3 pi/chat.py           # ikinci bir terminalde Türkçe sohbet
```

Açılışta otomatik başlatmak için `pi/robotai4b.service` dosyasındaki talimatlara bakın.
Robotun kendi kodundan kullanmak için OpenAI uyumlu uç noktaya istek atın:

```python
import json, urllib.request
body = {"messages": [{"role": "system", "content": "Senin adın RobotAI4B. ..."},
                     {"role": "user", "content": "Merhaba!"}]}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
```

## 2) LoRA ince ayarı (GPU gerekir)

Pi'de eğitim yapılamaz; eğitim bir kez GPU'lu bir makinede yapılır, sonuç Pi'ye indirilir.
En kolayı: **`notebooks/RobotAI4B_Colab.ipynb`** dosyasını Google Colab'da açın.

| GPU | Ayar | Not |
|---|---|---|
| A100 / L4 / RTX 3090-4090 (≥ 24 GB) | bf16 LoRA | Önerilen |
| T4 (16 GB, ücretsiz Colab) | `--load-in-4bit` (QLoRA) | Daha yavaş; birleştirme için RAM yetmeyebilir → Seçenek B |

Elle çalıştırma:

```bash
pip install -r requirements-train.txt
python scripts/prepare_data.py --config configs/robotai4b.yaml
python scripts/train_lora.py   --config configs/robotai4b.yaml      # T4 için: --load-in-4bit
```

Varsayılan veri karışımı (`configs/robotai4b.yaml`):

| Kaynak | Örnek | Açıklama |
|---|---|---|
| `data/robotai_persona.jsonl` | 48 × 4 tekrar | Robot kimliği, sınırlarını bilme, kısa cevap, güvenlik |
| `bezir/turkish_exam_instructions` | 3000 | Doğal Türkçe soru-cevap (veri kartında lisans belirtilmemiş) |
| `afkfatih/turkish-gemma-51k` | 3000 | Alpaca'nın Türkçe çevirisi, filtrelenmiş (Apache 2.0) |

> **Önemli:** Gemma 4 E4B'nin Türkçesi zaten iyi. Düşük kaliteli çeviri verisiyle uzun eğitim
> modeli **kötüleştirebilir**. Bu yüzden ayarlar bilerek hafif tutuldu (r=16, 1 epoch, lr=1e-4).
> LoRA'nın asıl kazandırdığı: robotun kimliği, üslubu ve kendi alanınızdaki bilgiler.
> En büyük iyileşme, `data/robotai_persona.jsonl` dosyasına **kendi robotunuza özel** yüzlerce
> iyi örnek eklemekle gelir. Eğitimden önce ve sonra `scripts/eval_tr.py` ile karşılaştırın.

## 3) Eğitilen modeli Pi'ye aktarma

**Seçenek A — Birleştirilmiş model (en hızlı çalışma, ≥ 25 GB RAM'li makinede):**

```bash
python scripts/merge_lora.py --adapter outputs/robotai4b-lora/final --out outputs/robotai4b-merged
bash scripts/export_gguf.sh outputs/robotai4b-merged outputs/gguf Q4_K_M Q4_0
# -> outputs/gguf/robotai4b-Q4_K_M.gguf  (+ robotai4b-mmproj.gguf: kamera/görüntü desteği)
```

Dosyayı Hugging Face'e yükleyin (Colab defterinin son hücresi) ve Pi'de:

```bash
MODEL_REPO=kullaniciadi/robotai4b-GGUF MODEL_FILE=robotai4b-Q4_K_M.gguf bash pi/setup_pi.sh
MODEL_FILE=robotai4b-Q4_K_M.gguf bash pi/run_server.sh
```

(Kalıcı yapmak için bu değerleri `pi/robotai4b.env` içine yazın. Depo özelse `HF_TOKEN=hf_...` ekleyin.)

**Seçenek B — Sadece LoRA adaptörü (az RAM yeterli, ~50-100 MB dosya):**

```bash
bash scripts/export_lora_gguf.sh outputs/robotai4b-lora/final outputs/gguf
# Pi'de: temel model unsloth'tan, adaptör sizin deponuzdan
LORA_REPO=kullaniciadi/robotai4b-GGUF LORA_FILE=robotai4b-lora-F16.gguf bash pi/setup_pi.sh
LORA_FILE=robotai4b-lora-F16.gguf bash pi/run_server.sh
```

## Performans ve bellek (Pi 5, 16 GB)

- **Bellek:** model ~4.6 GiB + 4096 token bağlam için birkaç yüz MB → toplam ~6 GB. 16 GB'ın
  büyük kısmı robotun diğer yazılımlarına kalır. (8 GB Pi'de de çalışır.)
- **Hız (tahmini):** 4 çekirdekli bir x86 sunucuda 8 token/sn ölçüldü. Pi 5'in bellek bant
  genişliği daha düşük olduğu için **saniyede yaklaşık 3-5 token** bekleyin (gerçek Pi'de ölçülmedi).
  Kendi Pi'nizde ölçmek için:
  `~/robotai4b/llama.cpp/build/bin/llama-bench -m ~/robotai4b/models/gemma-4-E4B-it-Q4_K_M.gguf -t 4`
- **Daha hızlı yapmak için:** `Q4_0` nicemlemesi ARM'de genellikle daha hızlıdır (`MODEL_FILE=gemma-4-E4B-it-Q4_0.gguf`);
  Pi'ye aktif soğutucu takın (yoksa işlemci ısınıp yavaşlar); cevapları kısa tutun.
- **Konuşan robot için:** cevabı akış (stream) olarak alıp cümle cümle ses sentezine verin
  (`pi/chat.py` akışlı okumanın örneğidir); böylece cevabın tamamını beklemeden birkaç saniye içinde konuşmaya başlar.

## Doğrulama

Bu depodaki hat (veri hazırlama → LoRA eğitimi → birleştirme → GGUF dönüştürme → nicemleme →
llama.cpp'de çalıştırma, ayrıca LoRA-GGUF yolu) Gemma 4 mimarisinin küçültülmüş bir kopyasıyla
uçtan uca test edildi. Gerçek 4B modelin eğitimi GPU gerektirdiği için burada çalıştırılmadı.

## Lisans

Kod: bu deponun lisansı. Gemma 4 ağırlıkları: Apache 2.0. Veri setlerinin lisansları için
ilgili Hugging Face sayfalarına bakın.
