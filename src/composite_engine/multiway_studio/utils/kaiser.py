def kaiser_beta_to_attenuation_db(beta):
    """Kaiser設計近似式からβに対応する目標阻止帯域減衰量を返す。"""
    beta = max(0.0, float(beta))
    if beta == 0.0:
        return 21.0

    linear_boundary_beta = 0.1102 * (50.0 - 8.7)
    if beta >= linear_boundary_beta:
        return beta / 0.1102 + 8.7

    low_db, high_db = 21.0, 50.0
    for _ in range(60):
        attenuation_db = (low_db + high_db) / 2.0
        delta_db = attenuation_db - 21.0
        estimated_beta = (
            0.5842 * delta_db ** 0.4
            + 0.07886 * delta_db
        )
        if estimated_beta < beta:
            low_db = attenuation_db
        else:
            high_db = attenuation_db
    return (low_db + high_db) / 2.0


def kaiser_attenuation_usage_hint(attenuation_db):
    """帯域分割用途としての短い減衰量目安を返す。"""
    attenuation_db = float(attenuation_db)
    if attenuation_db < 60.0:
        return "軽量・試聴向け（帯域外漏れを実測確認）"
    if attenuation_db < 75.0:
        return "通常再生の実用域"
    if attenuation_db < 90.0:
        return "標準（通常～大音圧）"
    if attenuation_db < 110.0:
        return "余裕重視（強い共振・保護）"
    return "非常に深い減衰（遷移幅とのバランス確認）"
