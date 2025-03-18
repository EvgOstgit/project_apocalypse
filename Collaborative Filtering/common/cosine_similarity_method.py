import numpy as np

# Функция для вычисления косинусного сходства между двумя пользователями
def cosine_similarity(user1, user2):
    # Ищем общие элементы (например, статьи, которые оба пользователя читали)
    mask = ~np.isnan(user1) & ~np.isnan(user2)
    if np.sum(mask) == 0:
        return 0  # Если нет общих оценок, сходство = 0
    user1_filtered = user1[mask]
    user2_filtered = user2[mask]
    
    # Косинусное сходство
    numerator = np.dot(user1_filtered, user2_filtered)
    denominator = np.linalg.norm(user1_filtered) * np.linalg.norm(user2_filtered)
    return numerator / denominator

# Функция для получения рекомендаций на основе похожих пользователей
def recommend_for_user(user_index, data, num_recommendations=2):
    similarity_scores = []
    
    for i in range(data.shape[0]):
        if i != user_index:
            similarity_scores.append((i, cosine_similarity(data[user_index], data[i])))
    
    # Сортируем пользователей по схожести
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Рекомендации для выбранного пользователя
    recommendations = []
    for user, similarity in similarity_scores:
        # Рекомендации из новостей, которые пользователю ещё не показаны
        for i in range(data.shape[1]):
            if data[user, i] == 1 and data[user_index, i] == 0:
                recommendations.append(i)
                if len(recommendations) >= num_recommendations:
                    return recommendations
    return recommendations

if __name__ == '__main__':
    # Пример матрицы, где 1 означает, что пользователь читал статью, а 0 — не читал
    data = np.array([
        [1, 1, 0, 1, 0],  # Пользователь 1
        [1, 0, 1, 1, 0],  # Пользователь 2
        [0, 1, 1, 0, 1],  # Пользователь 3
        [1, 1, 1, 0, 1]   # Пользователь 4
    ])
    # Пример: получить рекомендации для пользователя 0
    user_index = 2
    recommendations = recommend_for_user(user_index, data)
    print(f"Рекомендации для пользователя {user_index + 1}: новости {recommendations}")
