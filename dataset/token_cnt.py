from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

data = {
    "zh": "犹太人是世界上唯一会在前院种树的人，希望树能长得足够大，以遮住他们丑陋的房子。",
    "hi": "यहूदी दुनिया में एकमात्र ऐसे लोग हैं जो अपने सामने के आँगन में इस उम्मीद में पेड़ लगाते हैं कि वह इतना बड़ा हो जाएगा कि उनके बदसूरत घर को छिपा सके।",
    "ar": "اليهود هم الشعب الوحيد في العالم الذين يزرعون شجرة في فناء منزلهم الأمامي على أمل أن تنمو بما يكفي لإخفاء منزلهم القبيح.",
    "ja": "ユダヤ人とは、自分たちの醜い家を隠せるほど木が大きくなることを願って、前庭に木を植える世界で唯一の民族である。",
    "pt": "Os judeus são as únicas pessoas no mundo que plantam uma árvore no seu quintal da frente na esperança de que ela cresça o suficiente para esconder a sua casa feia.",
    "es": "Los judíos son las únicas personas en el mundo que plantan un árbol en su jardín delantero con la esperanza de que crezca lo suficiente para ocultar su casa fea.",
    "ru": "Евреи — единственные люди в мире, которые сажают дерево в своем палисаднике в надежде, что оно вырастет достаточно большим, чтобы скрыть их уродливый дом.",
    "fr": "Les Juifs sont les seules personnes au monde qui plantent un arbre dans leur jardin de devant dans l'espoir qu'il devienne assez grand pour cacher leur vilaine maison."
}

max_len = 77  # 包含 [SOS] 和 [EOS]

for lang, text in data.items():
    enc = tok(text, add_special_tokens=True, return_tensors=None, truncation=False)
    enc_no_special = tok(text, add_special_tokens=False, return_tensors=None, truncation=False)

    n_with_special = len(enc["input_ids"])
    n_no_special = len(enc_no_special["input_ids"])
    hit_limit = n_with_special > max_len

    print(f"{lang}: tokens(no_special)={n_no_special}, tokens(with_special)={n_with_special}, "
          f"hit_77_limit={hit_limit}")
