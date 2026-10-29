from scripts.score_lora_asr import cer, normalize, rank, summarize, term_hits


def test_normalize_ignores_punctuation_spacing_and_tai_variants():
    assert normalize("請先到一樓櫃檯掛號，記得帶健保卡。") == normalize(
        "請先到 一樓櫃台掛號 記得帶健保卡"
    )


def test_cer_counts_character_edits_against_reference_length():
    assert cer("哺乳室在二樓", "哺乳室在二樓") == 0.0
    assert cer("哺乳室在二樓", "柏露室在二樓") == 2 / 6


def test_term_hits_does_not_double_count_overlapping_terms():
    # 「健保卡」已計分，剩下的「健保」不能再從同一段文字裡算一次。
    assert term_hits(["健保卡", "健保"], "記得帶健保卡") == 1
    assert term_hits(["健保卡", "健保"], "記得帶健保卡，健保有給付") == 2


def _rows(group_cers):
    return [
        {"group": g, "cer": c, "hits": 0, "terms": 0}
        for g, c in group_cers
    ]


def test_rank_orders_by_target_cer_and_demotes_ctrl_regressions():
    summaries = {
        "base": summarize(_rows([("novel", 0.20), ("ctrl", 0.07)])),
        "good": summarize(_rows([("novel", 0.12), ("ctrl", 0.08)])),
        "better_but_breaks_ctrl": summarize(_rows([("novel", 0.05), ("ctrl", 0.20)])),
        "ok": summarize(_rows([("novel", 0.15), ("ctrl", 0.07)])),
    }

    ranked = rank(summaries)

    assert [r["tag"] for r in ranked] == ["good", "ok", "better_but_breaks_ctrl"]
    assert ranked[-1]["regressed"] is True
