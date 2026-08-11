# Strategy 3 — ASU'yu Geçme Planı

Bu doküman bir plan dokümanıdır, kod değildir. Amaç: takımın "hangi yöne
gidiyoruz" sorusunu netleştirmek ve kodlamaya başlamadan önce üzerinde
anlaşılması gereken 2 açık karar noktasını işaretlemek.

Bağlam: [`Strategy 1/CLAUDE.md`](../Strategy%201/CLAUDE.md),
[`Strategy 1/REPO_STUDY_NOTES.md`](../Strategy%201/REPO_STUDY_NOTES.md),
[`Strategy 1/PPO_PLUS_RULES.md`](../Strategy%201/PPO_PLUS_RULES.md) ve
`Strategy 2`'deki birebir aynı üç dosya. Bu plan onların kurallarını
değiştirmez, üzerine inşa eder.

## 0. Şu ana kadar ölçülmüş olan durum

| Policy | Kurulum | Sonuç |
|---|---|---|
| `asu_value_v1` | Seat-balanced, 100 oyun, Fixed-A/B/C'ye karşı | **72 ASU, 0 Fixed-A, 10 Fixed-B, 18 Fixed-C win** |
| PPO v2 (hybrid) | 2.000 oyun, Fixed-A/B/C'ye karşı | Final pencere **%0**, en iyi pencere **%2,5** |
| DDQN v2 (hybrid) | 2.000 oyun, Fixed-A/B/C'ye karşı | **0/200 win** |
| CFR-style rollout | 4 oyuncu, 1 oyun | 1.532 saniye (25dk32sn), 12.833 information set |

Kaynak: [`TRAINING_RESULTS.md`](../Strategy%202/TRAINING_RESULTS.md),
[`REPO_STUDY_NOTES.md`](../Strategy%202/REPO_STUDY_NOTES.md) §1 ve §5.

Üç sonuç çıkar:

1. **Saf PPO/DDQN RL zaten denenmiş ve başarısız** (0–%2,5). Aynı algoritmayı
   aynı hyperparametrelerle tekrar çalıştırmak yeni bir sonuç üretmez.
2. **ASU bir RL policy değil** — elle yazılmış bir heuristic (`asu_value_v1`,
   bkz. [`ASU_FROZEN_TEACHER/spec.py`](../Strategy%201/ASU_FROZEN_TEACHER/spec.py)).
   Şu an masadaki en güçlü oyuncu bu.
3. **CFR bu bütçede ölü**: 1 oyun 25 dakika. 5 günlük deadline'da bir CFR
   track'i anlamsız — bu yol elenmiştir, Strategy 3 kapsamı dışında.

Ölçülen 72/100 sayısı **`asu_value_v1`'e ait** (tek adımlık heuristic
değerlendirme, rollout yok). Daha güçlü varyant `asu_rollout_v1` (8 aday × 8
simülasyon × 32 adım lookahead, common random numbers — bkz.
[`REPO_STUDY_NOTES.md`](../Strategy%202/REPO_STUDY_NOTES.md) §5) repoda
**hiç ölçülmemiş**. "ASU'yu geçmek" hedefi bu dokümanda `asu_value_v1`'i
hedefler; `asu_rollout_v1`'i geçmek ayrı ve daha zor bir sonraki hedeftir.

## 1. Hedef metriğin tanımı — KARARLAŞTIRILDI: (b) baseline-relative

"ASU'yu geçmek" iki farklı şey anlamına gelebilirdi:

- (a) Head-to-head: aynı masada ASU'dan daha çok kazanmak.
- **(b) Baseline-relative (seçildi)**: eğitilen ajanın Fixed-A/B/C'ye karşı
  win rate'i, ASU'nun (`asu_value_v1`) aynı kurulumdaki **%72**'sini geçer.
  İki ajan aynı masaya oturmaz, aynı rakip setine karşı skorları
  karşılaştırılır.

**Birincil ve tek gate metriği (b)'dir.** Bu, mevcut 72/100 artifact'ıyla
doğrudan karşılaştırılabilir olduğu ve zaten repoda ölçülü olduğu için
seçildi. (a) head-to-head bu planın kapsamında **değil** — istenirse ayrı
bir sonraki adım olarak ele alınır.

Bu seçim §5 ve §7'deki arena/eval kurulumunu belirler: eğitilen policy,
ASU değil, Fixed-A/B/C'ye karşı test edilir; ASU sadece training'de
opponent/teacher rolünde kullanılır (§2), eval'de rakip değildir.

## 2. ASU'nun rolü — DÜZELTME: Seçenek A (sadece opponent koltuğu)

**Bu bölüm daha önce "Seçenek B, legal, kod doğruluyor" diyordu — o yanlıştı,
düzeltiliyor.** Gerekçe: `monopoly_bench.training.Trainer`, elinde checkpoint
olmayan her taze çalıştırmada `__init__` içinde otomatik olarak
`self._bootstrap()` çağırıyor
([`training.py:640-642`](monopoly_bench/training.py)) — bunu atlayacak
hiçbir CLI flag'i veya parametre yok; `--asu-expert-data` vermesen bile
`Trainer` kendi kendine ASU'yu oynatıp veri topluyor
([`training.py:650-666`](monopoly_bench/training.py)), sonra ağın policy
head'ini 2000 gradient adımı (`value_updates` varsayılanı) boyunca ASU'nun
seçtiği aksiyona karşı cross-entropy ile eğitiyor
([`training.py:712-724`](monopoly_bench/training.py), `bootstrap_asu_expert`).
`bootstrap_games`/`value_updates` gibi sayaçları sıfırlamaya çalışsan bile
`TrainingConfig.__post_init__` reddediyor (`min(counts) < 1` kontrolü).

"Ayrı bir faz, self-play değil, azalıyor" savım **ne zaman** sorusuna
cevaptı, kuralın sorduğu **olur mu** sorusuna değil. CLAUDE.md'nin Hard
Rules bölümü açık: "PPO, DDQN, CFR, ve MonopolyZero self-play gerçek oyun
sonucundan öğrenmeli, **ASU'nun seçtiği aksiyondan değil**", tek istisna
SLM/Gemma. Bootstrap tam olarak bu yasağı ihlal ediyor. Takımın kendi kural
mesajı da (bkz. `CLAUDE.md` "Competition rules") aynı çizgide: "ASU'ya karşı
oynayabilirsiniz ama ASU'yu birebir output klonlamak yasak."

Sonuç: **`monopoly_bench`/MonopolyZero self-play track'i tamamen bırakıldı.**
Kod değiştirilmedi (frozen v1 olarak işaretli, dokunmadık); sadece Strategy 3
pipeline'ından çıkarıldı. `monopoly_bench` dosyaları repoda referans için
duruyor ama `Trainer` public API üzerinden kurallara uygun çalıştırılamıyor.

### Seçenek A — sadece opponent koltuğu (şimdi tek seçenek)

Kullanıcı isteği: *"ASU'yu sadece opponent olarak, teacher rolünde
kullanarak nasıl geçebiliriz"*. Bu cümle iki okumaya açık, ikisi de kodda
zaten var ve **birbiriyle çelişmiyor** ama farklı güç/maliyet/risk profili
taşıyor. İkisi de burada isim verilerek karara bağlanmalı, sessizce
seçilmemeli:

### Seçenek A — sadece opponent koltuğu

ASU sadece training sırasında bir rakip olarak masaya oturur
(`asu_factory` / `asu_probability`,
[`train.py:64-104`](../Strategy%202/monopoly_game_engine/train.py)).
Öğrenen ajan hiçbir zaman ASU'nun seçtiği aksiyonu etiket olarak görmez —
sadece ASU'nun oynadığı oyunların **sonucundan** (kazandı/kaybetti) öğrenir.
Maliyet: ASU'nun karar maliyeti bir fixed-heuristic'in 10-100 katı, bu yüzden
kod zaten koltuk başına en fazla 1 ASU'ya izin veriyor ve olasılıkla
sınırlıyor.

- Artı: en katı okuma, imitasyon tavanı yok, "sadece opponent" ifadesini
  harfiyen karşılar.
- Eksi: mevcut Fixed-A/B/C'ye karşı %0 sonucu göz önüne alınca, öğrenmeyi
  hızlandıracak hiçbir sinyal eklemez — sadece rakip havuzunu güçlendirir.
  Keşif hâlâ sıfırdan.

### Seçenek B — opponent koltuğu + decaying bootstrap (REDDEDİLDİ)

Önceki halde buradaydı: AlphaZero'daki gibi kısa bir warm-start + self-play'in
öğretmeni aşabileceği savıyla önerilmişti. Sav kavramsal olarak yanlış değil
(AlphaGo/AlphaZero gerçekten böyle çalışıyor), ama bu repoda **uygulanabilir
değil** — `monopoly_bench.Trainer`'ın bootstrap'ı yukarıda açıklandığı gibi
kapatılamıyor, yani "kısa ve kontrollü" bir warm-start olarak sunulamıyor;
her çalıştırmada koşulsuz devreye giriyor. Uygulamak için `training.py`'a
bootstrap'ı atlanabilir yapan bir kod değişikliği gerekirdi — bu, "frozen v1"
olarak işaretlenmiş paylaşılan bir modülü değiştirmek demek, tek başına ayrı
bir karar ve bu planın kapsamı dışında bırakıldı.

### Netleştirilmesi gereken üçüncü şey: fine-tune kavramı yanlış yerde

İstekte geçen "fine tune" kelimesi bu repoda tek bir şeye karşılık geliyor:
`SLM_HANDMADE_MONOPOLY/` (Gemma 4 QLoRA). O yol **tamamen supervised
distillation** — ASU'nun etikettiği kararları taklit ediyor, RL değil
(bkz. `REPO_STUDY_NOTES.md` §10). Bu yüzden "sadece opponent" kısıtıyla
doğrudan çelişir ve ASU'nun gücünü **hiçbir zaman** aşamaz — imitasyonun
tavanı budur. Fine-tune/QLoRA, ASU'yu geçme hedefi için bir aday değil;
Strategy 3 kapsamı dışında bırakılıyor.

**Sonuç: Seçenek A.** ASU eğitimde sadece düşük olasılıklı bir opponent
koltuğu (`--asu-opponent-probability`, `train.py`'nin `_sample_opponents`'ı)
— hiçbir zaman bir öğrenme hedefi değil. Aşağıdaki bölümler bu düzeltilmiş
karar üzerinden güncellendi.

## 3. Kavram primer'i (kullanıcının istediği "önce öğrenmek")

Kararı doğru vermek için repodaki somut sayılarla:

- **Learning rate (LR)**: ağırlıkların her adımda ne kadar değişeceğini
  ölçekler. Mevcut DDQN v2 koşusu `1e-5` LR kullandı — bu, **10.000 oyunluk**
  paper koşusu için ayarlanmış bir değer, ama gerçekte **2.000 oyun**
  çalıştırıldı ([`PPO_PLUS_RULES.md:76-79`](../Strategy%202/PPO_PLUS_RULES.md)).
  Bu tek fark bile "neden %0" sorusunun somut bir parçası: çok küçük bir LR,
  kısa bütçede ağın anlamlı bir şey öğrenmesine yetecek kadar hareket
  etmiyor olabilir.
- **Fine-tune vs. RL**: fine-tune (burada QLoRA), dondurulmuş bir modele
  küçük, öğretilmiş bir düzeltme ekler — bir öğretmenin (ASU) etiketlerini
  taklit eder, tavanı öğretmenin gücüdür. RL (PPO/DDQN/self-play), oyunun
  gerçek kazan/kaybet sonucundan öğrenir, tavanı yoktur ama daha yavaş ve
  daha az örnek-verimlidir. Bu ikisi birbirinin yerine geçmez.
- **Model üretme (training loop)**: state → policy'nin aksiyon seçmesi →
  `env.step` → yeni state → reward. Bu döngü binlerce/milyonlarca kez
  tekrarlanır, ağırlıklar her batch'te güncellenir. Repoda bu döngü
  [`train.py`](../Strategy%202/monopoly_game_engine/train.py)'de.
- **Hyperparametre değiştirme**: LR, discount (`gamma=0.9999`), batch size
  (128), replay buffer (10.000), target-sync interval (500 oyun) — hepsi
  `PPO_PLUS_RULES.md`'de sabit ve **10.000 oyunluk bütçe için** seçilmiş.
  Strategy 3'ün somut işlerinden biri, bunları gerçek 1.000–3.000 oyunluk
  bütçeye göre yeniden ayarlamak (örn. daha yüksek LR, daha kısa target-sync,
  daha küçük replay).

### DDQN retuning — kararlaştırıldı, Colab'da doğrulanacak

`tools/train_and_save.py` zaten bu ayarlar için CLI override sağlıyor
(`--lr`, `--epsilon-decay`, `--target-update-freq-steps`), kod içi
yorum satırı bile teşhisi taşıyor: 1e-5 LR + 500-oyunda-1-sync kombinasyonu
1000 oyunda 2 senkronla %0'da kaldı. Seçilen başlangıç değerleri:

| Parametre | Eski (10k-oyun için) | Yeni (1-2k bütçe için) | Neden |
|---|---|---|---|
| `--lr` | `1e-5` | `1e-4` | 10 kat daha büyük adım; kısa bütçede ağın anlamlı hareket etmesi için. Önerilen aralığın (1e-4–1e-3) alt ucu, ilk deneme için daha güvenli. |
| `--epsilon-decay` | `0.9995` (floor ~oyun 6000) | `0.9985` (floor ~oyun 2000) | Keşif fazı bütçeye sığmalı; aksi halde koşu bitene kadar hâlâ çoğunlukla rastgele hamle yapılır. |
| `--target-update-freq-steps` | yok (sadece 500 oyunda 1) | `200` (gradient adımı, oyun-bazlı senkrona ek) | Target ağ, online ağın öğrendiğini çok geç görüyordu (1000 oyunda sadece 2 senkron); adım-bazlı senkron bunu sıklaştırır. |

**Doğrulama durumu**: 20 oyunluk yerel smoke test denendi
(`--games 20 --device cpu`), komut `0.27 GiB` RSS'te, flag'ler hatasız
kabul edildi, ama memory watchdog makinenin o an boş RAM'i `2 GiB` eşiğinin
altına düştüğü için (`vm_stat` ile doğrulandı, gerçek bir kaynak kısıtı,
kod hatası değil) 0 oyunda durdurdu. `CLAUDE.md` kuralı gereği bu eşiği
büyük bir koşuyu yerelde zorlamak için yükseltmedim. Sonuç: mekanik olarak
flag'ler çalışıyor, gerçek 1.000–2.000 oyunluk doğrulama Colab'da yapılacak
(zaten proje kuralı bunu gerektiriyor).
- **`c_puct` ve progressive widening** (sadece Seçenek B / MonopolyZero
  yolunda): `c_puct`, search'ün "bilinen iyi aksiyonu tekrar dene" ile
  "az denenmiş aksiyonu keşfet" arasındaki dengesini ayarlar. Progressive
  widening, 2.958 aksiyonun hepsini her node'da açmak yerine önce küçük bir
  alt kümeyi açıp visit count arttıkça genişletir — aksi halde trade
  ailesindeki 2.268 exchange aksiyonu search'ü boğar
  (`REPO_STUDY_NOTES.md` §9).

## 4. Neden PPO/DDQN retry değil aynı ayarlarla — ve §2'nin kısıtladığı şey

`CLAUDE.md`, %0 sonucunun 4 sebebini teşhis etmiş. Her biri için karşı-önlem,
§2'nin düzeltilmesinden sonraki gerçek durum:

| Teşhis edilen sebep | Karşı-önlem | Durum |
|---|---|---|
| Seyrek, gecikmeli reward, uzun horizon | Search (PUCT), lookahead değeri | **Yok** — `monopoly_bench`/search §2'de elendi. Karşılığı yok. |
| 2.958 aksiyonun 2.268'i trade-exchange, uzayı domine ediyor | Section-balanced exploration | **Var** — DDQN'de zaten kodlu (`REPO_STUDY_NOTES.md` §6), `monopoly_bench`'e bağlı değil. |
| Opponent non-stationarity (3 sabit rakip) | Self-play snapshot pool | **Var** — `monopoly_game_engine.self_play.SelfPlayPool` + `--self-play-probability` (DDQN, `train_and_save.py`). ASU'ya hiç dokunmuyor, tamamen legal. |
| Hyperparametreler 10.000 oyun için, bütçe 1.000–2.000 | Yeniden ayarlanmış LR/target-sync | **Var** — §3'teki DDQN retuning. |

Saf PPO/DDQN'i **eski** hyperparametrelerle tekrar çalıştırmak hâlâ önerilmiyor
(tablonun son satırı). Ama dürüst olmak gerekirse: search/lookahead
karşı-önlemi (ilk satır) artık elimizde değil — bu, §2'nin ASU-bootstrap'ı
reddetmesinin gerçek maliyeti. Kalan üç önlemle (retuned hyperparametreler +
self-play pool + section-balanced exploration) 2.000 oyunda %0'dan daha
iyisini yapmak hedef, ama en güçlü teorik karşı-önlemi kaybettik.

Kullanılan algoritmalar açık kaynaklı ve yayınlanmış: PPO
([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)), Double DQN
([arXiv:1509.06461](https://arxiv.org/abs/1509.06461)). Kapalı kaynak
bağımlılık yok, hepsi bu repodaki kendi Python koduyla implemente edilmiş.

## 5. Pipeline (Seçenek A)

```text
1. PPO/DDQN eğitimi, retuned hyperparametreler (§3) +
   self-play snapshot pool (opponent non-stationarity için) +
   ASU opponent koltuğu, düşük olasılık (--asu-opponent-probability)
                          -- ASU'nun hiçbir aksiyonu hiçbir zaman hedef değil
                          -- sadece gerçek oyun sonucundan öğrenir
2. Değerlendirme: Fixed-A/B/C'ye karşı, seat-balanced, Wilson interval
                          -- asu_value_v1'in 72/100'üyle karşılaştırma (§1)
```

`monopoly_bench` (arama/self-play/MonopolyZero) ve bootstrap fazı bu
pipeline'da yok — §2'de gerekçesiyle reddedildi.

## 6. Reuse kararı — KARARLAŞTIRILDI ve UYGULANDI

`monopoly_bench/` (MonopolyZero: `search.py`, `training.py`, `model.py`,
`ladder.py`) ve `ASU_FROZEN_TEACHER/` sadece **Strategy 1**'de vardı.
`Strategy 3` şu şekilde kuruldu:

- **Taban**: `Strategy 2`'nin tamamı (`monopoly_game_engine/`, `tools/`,
  `tests/`, `training_guard.py`, `colab/`, doküman dosyaları) — Strategy 1
  ile içerik olarak birebir aynı doğrulandı (`diff -rq` farksız), Strategy 2
  "current PPO-focused work" olarak seçildi.
- **Eklenen**: `Strategy 1/ASU_FROZEN_TEACHER/` ve `Strategy 1/monopoly_bench/`
  aynen kopyalandı (SLM/Gemma ve CFR track'leri hariç — §0'da elendi,
  kapsam dışı).
- **Doğrulama**: Python 3.14 ile import zinciri denendi
  (`monopoly_bench` → `ASU_FROZEN_TEACHER` → `monopoly_game_engine`).
  Modül yolları doğru çözülüyor; `numpy`/`torch` bu makinede kurulu
  olmadığı için tam çalıştırma denenmedi — **repoda hiçbir yerde
  `requirements.txt`/`pyproject.toml`/venv yok**, bu ortam kurulumu ayrı,
  ele alınmamış bir konu, plan onayına dahil değil.

## 7. Değerlendirme protokolü (kod yazılmadan önce sabitlenmeli)

`REPO_STUDY_NOTES.md` §11 kuralına göre, her sonuç şu bilgilerle
raporlanır:

- **Rakip kimliği**: ASU (`asu_value_v1`, hangi noise/epsilon ile) + hangi
  Fixed personality'ler (A-F).
- **Koltuk dengesi**: öğrenilen policy her fiziksel koltukta eşit sayıda
  oynar.
- **Paired seed**: karşılaştırılan policy'ler aynı zar/şans akışını görür.
- **Round cap**: 200, tiebreak simulator net worth.
- **Belirsizlik**: küçük örneklem için Wilson interval — bare yüzde yok.
- **Checkpoint kimliği**: `ppo-plus-v2` ruleset/state/action hash kontrolü.

## 8. Başarı kriteri ve gerçekçi beklenti

- **Minimum bar**: mevcut %0'ın istatistiksel olarak anlamlı üzerinde bir
  win rate, Fixed-A/B/C'ye karşı (§1b).
- **Hedef**: `asu_value_v1`'in 72/100'ünü (Fixed-A/B/C'ye karşı) geçmek.
- **Stretch, bu planın kapsamı dışında**: `asu_rollout_v1`'i geçmek —
  repoda hiç ölçülmemiş, muhtemelen daha güçlü bir hedef.
- 5 günlük, çoğunlukla CPU-yerel + Colab-ağır bütçede, **ve search/lookahead
  karşı-önlemi olmadan** (§4), `asu_value_v1`'in 72'sini geçmek daha da
  belirsiz hale geldi — bu risk açıkça kabul ediliyor, gizlenmiyor.

## 9. Karar durumu

1. §1 Hedef metriği: **(b) baseline-relative** — karara bağlandı, değişmedi.
2. §2 ASU rolü: **Seçenek A** (sadece opponent koltuğu) — önce yanlışlıkla
   Seçenek B "legal" denip onaylanmıştı, `monopoly_bench.Trainer`'ın
   koşulsuz bootstrap'ı fark edilince ve takımın kendi kural mesajıyla
   (`CLAUDE.md`, "Competition rules") doğrulanınca A'ya düzeltildi.
3. §6 Reuse: `ASU_FROZEN_TEACHER/` hâlâ kullanılıyor (opponent koltuğu +
   `evaluate_lineup` ile değerlendirme). `monopoly_bench/` kopyalandı ama
   Strategy 3 pipeline'ında **kullanılmıyor** — kod referans için duruyor.

## 10. Uygulama durumu

- **DDQN retuning**: `colab/PPO_DDQN_Train_Colab.ipynb` zaten §3'teki tam
  değerlerle geliyordu (`lr=1e-4`, `target-update-freq-steps=2000`,
  `epsilon-decay=0.9985`).
- **`colab/Strategy3_Train_Colab.ipynb`**: PPO + DDQN eğitimi
  (`--asu-opponent-probability 0.15`, opponent koltuğu) → `tools/evaluate_vs_fixed.py`
  ile Fixed-A/B/C'ye karşı nihai ölçüm. `collect-asu`/`monopoly_bench train`
  adımları çıkarıldı (§2 düzeltmesi).
- **`tools/evaluate_vs_fixed.py`**: `ASU_FROZEN_TEACHER.evaluate.evaluate_lineup`
  üzerine yazıldı — asu_value_v1'in 72/100'ünü **ürettiği kodun aynısı**,
  `ppo:`/`ddqn:` checkpoint spec'i destekliyor. Eğitilmemiş bir PPO
  checkpoint'iyle 4 oyunluk smoke test ile mekanik olarak doğrulandı.
- **Push**: `feature/strategy-3-hybrid` dalı, fork'a (`Gokturkakman/DeepRL_Monopoly`)
  push edildi. Bu düzeltmeler de aynı dala push edilecek.
