import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('user_data_ex.csv')  # 데이터 가져오기
df['keyword'] = df['keyword'].fillna('')
df['want_keyword'] = df['want_keyword'].fillna('')
df['match_record'] = df['match_record'].fillna('')

def similar_rank(keyword1, keyword2, df):
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix1 = vectorizer.fit_transform(df[keyword1])
    tfidf_matrix2 = vectorizer.fit_transform(df[keyword2])

    # 코사인 유사도 계산
    cosine_sim = linear_kernel(tfidf_matrix1, tfidf_matrix2)

    return cosine_sim

# id(index)랑 순서 넘버만 있는 데이터
indices = pd.Series(df.index, index = df['id'])

def get_recommendations(id) :

    # user는 유저 id에 해당하는 row만 추출한 데이터
    user = df[df['id'] == id].iloc[0]

    # 유저가 원하는 성별, 연령대에 맞는 사람들의 집합 = new_df
    df_sex = user['other_sex']
    df_age = user['other_age']
    new_df = df[(df['sex'] == df_sex) & (df['age'] == df_age)]

    # match_record가 비어있으면 want_keyword로 유사도 측정
    match_record = user['match_record']
    csm = 0

    if user['match_record'] == '':
        csm = similar_rank('want_keyword', 'keyword', new_df)

        #id를 통해 그 유저의 index 값 얻기
        idx = indices[id]

        #코사인 유사도 매트릭스에서 (idx, 유사도) 얻기
        sim_scores = list(enumerate(csm[idx]))

        #본인 삭제
        del sim_scores[idx]

        #코사인 유사도 기준으로 내림차순 정렬
        sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

        #5명의 추천인을 슬라이싱
        sim_scores = sim_scores[0:5]

        #추천인 목록 5명의 인덱스 정보 추출
        match_indices = [i[0] for i in sim_scores]

        #인덱스 정보를 통해 추천인 목록 추출
        rec_list = df['id'].iloc[match_indices]

    # match _record가 존재하는 경우 5점에게 가중치
    else :
        data_list = json.loads(match_record)

        csm = similar_rank('keyword', 'keyword', new_df)

        # 5점인 사람들의 id
        id_score5 = [data["id"] for data in data_list if data["score"] == 5]
        recommendation_set = set()

        # 5점인 사람들과 유사한 사람들 찾기 (중복 x 위해 set으로 찾음)
        for i in id_score5 :
            idx = indices[i]
            sim_scores_5 = list(enumerate(csm[idx]))
            del sim_scores_5[idx]
            sim_scores_5 = sorted(sim_scores_5, key = lambda x: x[1], reverse = True)
            sim_scores_5 = sim_scores_5[0:5]
            for i in sim_scores_5 :
                match_indices = i[0]
                recommendation_set.add(match_indices)

        csm = similar_rank('want_keyword', 'keyword', new_df)

        #id를 통해 그 유저의 index 값 얻기
        idx = indices[id]

        #코사인 유사도 매트릭스에서 (idx, 유사도) 얻기
        sim_scores = list(enumerate(csm[idx]))

        #sim_scores는 튜플 리스트, 따라서 유사도 값 변경이 불가하기 때문에 튜플 리스트 새로 만듦
        new_scores = []

        for i in range(len(sim_scores)) :
            if i in recommendation_set :
                new_scores.append((i, sim_scores[i][1] * 1.2))
            else :
                new_scores.append((i, sim_scores[i][1]))

        #id를 통해 그 유저의 index 값 얻기
        idx = indices[id]

        #본인 삭제
        del new_scores[idx]

        #코사인 유사도 기준으로 내림차순 정렬
        new_scores = sorted(new_scores, key = lambda x: x[1], reverse = True)

        #5명의 추천인을 슬라이싱
        sim_scores = new_scores[0:5]

        #추천인 목록 5명의 인덱스 정보 추출
        match_indices = [i[0] for i in sim_scores]

        #인덱스 정보를 통해 추천인 목록 추출
        rec_list = df['id'].iloc[match_indices]
        
    return rec_list

id = 'yoona126'
get_recommendations(id)
