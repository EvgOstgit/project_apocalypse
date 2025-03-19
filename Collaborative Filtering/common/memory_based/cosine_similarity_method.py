import numpy as np
import pandas as pd
import os

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

def calculate_user_similarity(df):
    similarity_matrix = np.zeros((df.shape[0], df.shape[0]))  # Создаем пустую матрицу схожести
    
    for i in range(df.shape[0]):
        for j in range(i + 1, df.shape[0]):  # Ищем только уникальные пары (i, j)
            similarity = cosine_similarity(df.iloc[i], df.iloc[j])  # Вычисляем сходство
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Симметричная матрица
            
    return pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# Функция для получения рекомендаций на основе похожих пользователей
def recommend_for_user(user_index, similarity_matrix, user_news_matrix, num_recommendations=2):
    similarity_scores = []

    # Получаем схожесть с другими пользователями из матрицы сходства
    user_similarities = similarity_matrix.iloc[user_index]  # Получаем схожесть выбранного пользователя с другими

    # Сортируем пользователей по схожести
    similarity_scores = user_similarities.drop(user_index).sort_values(ascending=False)

    # Рекомендации для выбранного пользователя
    recommendations = []
    for similar_user, similarity in similarity_scores.items():
        # Просматриваем новости, которые похожий пользователь кликал, но которые выбранный пользователь еще не видел
        for i in range(user_news_matrix.shape[1]):
            # Если новость была прочитана похожим пользователем, но не прочитана текущим пользователем
            if user_news_matrix.loc[similar_user, i] == 1 and user_news_matrix.loc[user_index, i] == 0:
                recommendations.append(i)
                if len(recommendations) >= num_recommendations:
                    return recommendations

    return recommendations



if __name__ == '__main__':

    # Получаем путь к каталогу memory_based
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Находим путь к родительскому каталогу common
    COMMON_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

    # Формируем путь к каталогу с данными
    DATA_DIR = os.path.join(COMMON_DIR, 'data\\Mindsmall_train')

    behaivors_path = os.path.join(DATA_DIR, 'behaviors.tsv')
    news_path = os.path.join(DATA_DIR, 'news.tsv')

    df = pd.read_csv(behaivors_path, sep='\t', header=None, names=['id', 'user_id', 'timestamp', 'shown_news', 'clicked_news'])

    # Пока достаточно использования только этих столбцов
    df = df[['user_id', 'clicked_news']]

    # Преобразуем список кликнутых новостей в массив
    df['clicked_news'] = df['clicked_news'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Получаем список пользователей
    user_ids = df['user_id'].unique()
    
    news_ids = list(set(news for news_list in df['clicked_news'] for news in news_list))

    # Создаем user-news матрицу
    user_news_matrix = pd.DataFrame(0, index=user_ids, columns=news_ids)

    # Заполнение матрицы (1 - если кликнул новость)
    for _, row in df.iterrows():
        user_news_matrix.loc[row['user_id'], row['clicked_news']] = 1

    user_similarity = calculate_user_similarity(user_news_matrix)

    print(recommend_for_user('U8125', user_similarity, user_news_matrix, 1))