import pandas as pd
import numpy as np


class Recommend():
    def __init__(self,similarity_matrix_dir):
        self.k = 15
        self.recommend_num = 30
        self.similarity_matrix = pd.read_csv(similarity_matrix_dir, header=0, index_col=0)
        # 列名读进来自动前边补个0 去掉
        columns = list(self.similarity_matrix.columns)
        for i in range(len(columns)):
            if columns[i][0] == '0':
                columns[i] = columns[i][1:]
        # 行名float->str
        index = self.similarity_matrix.index
        index = [str(i) for i in list(index)]
        self.similarity_matrix.columns = columns
        self.similarity_matrix.index = index

    def get_item_interest(self, j, user_like_set, j_top_k_set):
        # 获得用户对于j物品的兴趣 top_k是j相似度的top_k
        intersection_set = user_like_set & j_top_k_set
        interest = 0
        for i in intersection_set:
            interest += self.similarity_matrix.loc[i, j]
        return interest

    def get_recommend_list(self, user_like_set):
        items_list = list(self.similarity_matrix.columns)
        items_interest_list = pd.Series()
        for j in items_list:
            j_set = self.similarity_matrix.loc[j, :]
            #  获得和J最相近的k个物品的集合
            j_top_k_set = set(list(j_set.nlargest(self.k).index))
            j_interest = self.get_item_interest(j, user_like_set, j_top_k_set)
            items_interest_list[j] = j_interest
        return items_interest_list.nlargest(self.recommend_num)




if __name__ == '__main__':
    '''test_set_i = set(['20202', '81383', '20201', '10101', '40402', '30314', '30312', '30311', '50507', '30313'])
    test_set_j = set(['30314', '30312', '30311', '50507', '30313', '60601', '40401', '50514', '81474', '30310'])
    recommend_obj = Recommend('./tables/similarity_matrix.csv')
    recommend_list = recommend_obj.get_recommend_list(test_set_i)'''
