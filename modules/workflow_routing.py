def score_complexity(row, prob):
    score = 0
    if prob > 0.5:
        score += 50
    if row.get('policy_deductable', 0) > 1000:
        score += 20
    if row.get('policy_annual_premium', 0) > 1000:
        score += 10
    if row.get('number_of_vehicles_involved', 0) > 1:
        score += 10
    if row.get('witnesses', 0) > 0:
        score += 10
    return score