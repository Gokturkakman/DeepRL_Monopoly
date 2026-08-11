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

## 2. ASU'nun rolü — KARARLAŞTIRILDI: Seçenek B (opponent + decaying bootstrap)

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

### Seçenek B — opponent koltuğu + decaying bootstrap (önerilen)

A'ya ek olarak, self-play başlamadan önce ayrı bir offline ısınma fazı:
ağın **policy head'i** ASU'nun seçtiği aksiyonu cross-entropy ile taklit
eder, ama **value head'i her zaman gerçek oyun sonucunu** öğrenir — ASU'nun
kendi heuristic skorunu değil
([`monopoly_bench/training.py`](../Strategy%201/monopoly_bench/training.py)
`collect_asu_examples` / `bootstrap_asu_expert`). Bu ağırlık 8 nesil
boyunca sıfıra söner; self-play tamamen gerçek sonuçlardan öğrenir.

Bu, `CLAUDE.md`'nin "self-play ASU çıktısından öğrenemez" kuralıyla
çelişmiyor çünkü bootstrap self-play'in bir parçası değil, self-play
başlamadan önceki ayrı ve azalan bir fazdır. **Neden önerilen**: bir
imitasyon-sadece policy, matematiksel olarak öğretmeninin (ASU'nun)
gücünü aşamaz — kopyaladığı şeyin tavanı budur. AlphaZero'nun insan
oyunundan güçlü olmasının sebebi de tam olarak bu: kısa bir warm-start +
self-play + search, öğretmeni **aşabilen** tek bilinen mekanizma. Kod bu
yüzden zaten decay'i içeriyor — sonsuza kadar ASU'yu taklit etmek hedefte
değil.

- Artı: sıfırdan keşif yerine hızlı, makul bir başlangıç noktası. 5 günlük
  bütçede bu fark kritik olabilir (PPO/DDQN'in 2.000 oyunda hâlâ %0
  olduğunu unutmayın).
- Eksi: "sadece opponent" ifadesinin en dar okumasını ihlal eder — bootstrap
  aşamasında ASU'nun aksiyonu gerçekten bir hedef olarak kullanılıyor
  (decaying da olsa).

### Netleştirilmesi gereken üçüncü şey: fine-tune kavramı yanlış yerde

İstekte geçen "fine tune" kelimesi bu repoda tek bir şeye karşılık geliyor:
`SLM_HANDMADE_MONOPOLY/` (Gemma 4 QLoRA). O yol **tamamen supervised
distillation** — ASU'nun etikettiği kararları taklit ediyor, RL değil
(bkz. `REPO_STUDY_NOTES.md` §10). Bu yüzden "sadece opponent" kısıtıyla
doğrudan çelişir ve ASU'nun gücünü **hiçbir zaman** aşamaz — imitasyonun
tavanı budur. Fine-tune/QLoRA, ASU'yu geçme hedefi için bir aday değil;
Strategy 3 kapsamı dışında bırakılıyor.

**Seçenek B onaylandı, koşul "legalse" idi — legal, kod bunu doğruluyor:**
bootstrap `collect_asu_examples`/`bootstrap_asu_expert`
([`monopoly_bench/training.py`](monopoly_bench/training.py)) self-play'in
kendisi değil, ondan önceki ayrı ve 8 nesilde sıfıra sönen bir fazdır;
value head bootstrap sırasında bile gerçek kazananı öğrenir, ASU'nun
heuristic skorunu değil. Bu, `CLAUDE.md`'nin "self-play ASU çıktısından
öğrenemez" kuralını ihlal etmez çünkü kural self-play'i hedefler,
bootstrap'i değil — aynı dosyadaki `_sample_opponents` yorumu bu ayrımı
doğruluyor. Aşağıdaki bölümler B varsayımıyla yazıldı.

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

## 4. Neden PPO/DDQN retry değil, MonopolyZero-tarzı hybrid

`CLAUDE.md`, %0 sonucunun 4 sebebini teşhis etmiş. Her biri için somut
karşı-önlem:

| Teşhis edilen sebep | Karşı-önlem (Strategy 3) |
|---|---|
| Seyrek, gecikmeli reward, uzun horizon | Search (PUCT), sadece model-free bootstrap yerine her adımda lookahead değeri kullanır |
| 2.958 aksiyonun 2.268'i trade-exchange, uzayı domine ediyor | Progressive widening + section-balanced sampling (zaten `monopoly_bench`'te var) |
| Opponent non-stationarity (3 sabit rakip) | Self-play + snapshot pool, rakip sabit kalmaz, ajanla birlikte güçlenir |
| Hyperparametreler 10.000 oyun için, bütçe 1.000–2.000 | §3'teki gibi yeniden ayarlanmış LR/target-sync/replay + kısa curriculum |

Saf PPO/DDQN'i aynı hyperparametrelerle tekrar çalıştırmak bu tablodaki
hiçbir satırı değiştirmez — bu yüzden önerilmiyor. `monopoly_bench`
(MonopolyZero) bu 4 önlemi zaten kısmen kodlamış durumda, ama **sadece
Strategy 1'de var, Strategy 2'de yok** — bu yüzden §6'da açık bir
copy/reuse kararı gerekiyor.

Algoritma ailesi açık kaynaklı ve yayınlanmış: AlphaZero
([arXiv:1712.01815](https://arxiv.org/abs/1712.01815)), PPO
([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)). Kapalı kaynak
bağımlılık yok, hepsi bu repodaki kendi Python koduyla implemente edilmiş.

## 5. Önerilen pipeline (Seçenek B varsayımıyla)

```text
1. PPO warm start        -- mevcut PPO actor/critic ağırlıkları başlangıç noktası
2. ASU bootstrap (teacher rolü, decaying, 8 nesil)
                          -- policy head ASU aksiyonunu taklit eder
                          -- value head HER ZAMAN gerçek kazananı öğrenir
3. Self-play + snapshot pool + Fixed A-F + ASU (opponent rolü, düşük olasılık)
                          -- artık hiçbir ASU aksiyonu hedef değil
                          -- sadece gerçek oyun sonucu
4. Arena: aday vs. incumbent vs. ASU vs. Fixed A-F
                          -- istatistiksel + güvenlik gate'i geçmeden promote yok
```

Seçenek A seçilirse adım 2 tamamen atlanır, adım 1 de opsiyoneldir.

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
  head-to-head win rate (§1a).
- **Hedef**: `asu_value_v1`'i head-to-head geçmek.
- **Stretch, bu planın kapsamı dışında**: `asu_rollout_v1`'i geçmek —
  repoda hiç ölçülmemiş, muhtemelen daha güçlü bir hedef.
- 5 günlük, çoğunlukla CPU-yerel + Colab-ağır bütçede `asu_value_v1`'i
  kesin olarak geçmek garanti edilemez; bu risk açıkça kabul ediliyor,
  gizlenmiyor.

## 9. Karar durumu

1. §1 Hedef metriği: **(b) baseline-relative** — karara bağlandı.
2. §2 ASU rolü: **Seçenek B** (opponent + decaying bootstrap) — karara
   bağlandı, kod üzerinden legal olduğu doğrulandı.
3. §6 Reuse: **onaylandı ve uygulandı** — `Strategy 3` klasörü kuruldu.

Üç karar da kapandı. Sıradaki adım: §3-§5'teki hyperparametre yeniden
ayarı ve bootstrap/self-play kodunu `Strategy 3` içinde uyarlamak — bu
ayrı bir implementasyon adımı, bu dokümanın kapsamı dışında.

## 10. Uygulama durumu

- **DDQN retuning**: `Strategy 2`'den kopyalanan `colab/PPO_DDQN_Train_Colab.ipynb`
  zaten §3'teki tam değerlerle geliyordu (`lr=1e-4`,
  `target-update-freq-steps=2000`, `epsilon-decay=0.9985`) — ek iş gerekmedi.
- **Strategy 3'e özgü Colab notebook'u**: `colab/Strategy3_Hybrid_Train_Colab.ipynb`
  yazıldı. Sıra: PPO warm start (`--asu-opponent-probability 0.02` ile ASU
  opponent rolünde) → `python -m monopoly_bench collect-asu` (ASU teacher
  bootstrap dataset'i) → `python -m monopoly_bench train` (decaying bootstrap +
  self-play generations) → `tools/evaluate_vs_fixed.py` (nihai metrik).
- **`tools/evaluate_vs_fixed.py`**: yeni yazıldı — §1'de kararlaştırılan
  baseline-relative metriği (`monopoly_bench.arena` ile seat-balanced win rate
  + Wilson lower bound, Fixed-A/B/C'ye karşı) hesaplıyor. Mekanik olarak
  yerelde eğitilmemiş bir modelle 4 oyunluk smoke test ile doğrulandı
  (0/4 win, beklenen — eğitim yok).
- **Colab'da açma**: notebook henüz sadece yerelde var, fork'a (
  `Gokturkakman/DeepRL_Monopoly`) push edilmeden Colab'ın clone hücresi
  `Strategy 3`'ü bulamaz. Push, kullanıcı onayı gerektiren bir git işlemi —
  bu dokümanın/oturumun kapsamında otomatik yapılmadı.
