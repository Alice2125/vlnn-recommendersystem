import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
import warnings
import re
from surprise import Reader, Dataset, BaselineOnly
from surprise.model_selection.validation import cross_validate
from wordcloud import WordCloud
import plotly.graph_objs as go
import plotly.express as px

# 1. Read data
products = pd.read_parquet("products.parquet.gzip")
reviews = pd.read_parquet("reviews.parquet.gzip")


# 2. Data pre-processing

# 3. Build model
# Content-based filtering
products['name_description_wt'] = products['name_description'].apply(lambda x: word_tokenize(x, format = 'text'))
products = products.reset_index()
# stop words
STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE,'r', encoding="UTF-8") as file:
    stop_words = file.read()
    
stop_words = stop_words.split('\n')

# Dùng TFIDF
tfidf = TfidfVectorizer(analyzer = 'word', min_df=0, stop_words = stop_words)
tfidf_name_des = tfidf.fit_transform(products.name_description_wt)

content_based = cosine_similarity(tfidf_name_des, tfidf_name_des)

contentbased_solution1 = pd.read_csv('ContentBased_Solution1.csv', encoding='utf-8')
@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(contentbased_solution1)

    
@st.cache   
def name(item_id):
    return products.loc[products['item_id'] == item_id]['name'].to_list()[0].split('-')[0]
@st.cache(suppress_st_warning=True) 
def recommender(item_id, rec_num):
    st.code("Đề xuất " + str(rec_num) +' sản phẩm tương tự với sản phẩm '+ name(item_id) + ':')
    results = {}
    for idx, row in products.iterrows():    
        similar_indices = content_based[idx].argsort()[:-22:-1]
        similar_items = [(content_based[idx][i], products['item_id'][i]) for i in similar_indices]
        results[row['item_id']] = similar_items[1:]
    recommend = results[item_id][:rec_num]
    for rec in recommend:
        st.write('*'*10)
        st.code('Đề xuất : product_id:' + str(rec[1]) + '-' + name(rec[1]) + " với mức tương đồng là " + str(rec[0]))

# Collaborative filtering

reader = Reader()
data = Dataset.load_from_df(reviews[['customer_id','product_id','rating']], reader)
collaborative = BaselineOnly()
trainset = data.build_full_trainset()
collaborative.fit(trainset)

@st.cache
def BaselineOnly(customerId, bottom_rating):
    review_score = reviews[['product_id']]
    review_score['EstimateScore'] = review_score['product_id'].apply(lambda x: collaborative.predict(customerId, x).est)
    review_score = review_score.sort_values(by=['EstimateScore'], ascending=False)
    review_score = review_score.drop_duplicates()
    print("10 sản phẩm đề xuất cho customer_id "+ str(customerId) + ' là:')
    print(review_score.head(10))
    
# 4. Evaluate model
# Content based filtering
        

# def word_cloud(item_id, rec_num):
#     tst = {}
#     for idx, row in products.iterrows():    
#         similar_indices = content_based[idx].argsort()[:-22:-1]
#         similar_items = [(content_based[idx][i], products['item_id'][i]) for i in similar_indices]
#         tst[row['item_id']] = similar_items[1:]
#     id = [r[1] for r in tst[item_id]]
#     text = (products[products.item_id.isin(id)])
#     return ' '.join(text.name_description)

# def wc_draw(word_cloud):
#     wc = WordCloud(stopwords=stop_words).generate(word_cloud)
#     return plt.imshow(wc)

# Collaborative filtering
# result_B = cross_validate(collaborative, data, measures=['RMSE','MAE'], cv=3, verbose= True)

    
#--------------
# GUI
st.title("Data Science Project 2 - Recommender System")
st.write("## Tiki Products and Reviews Recommendation")
menu = ['Business Objective','Build Project','Recommendation for User']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
    ##### I. Tổng quan về hệ thống gợi ý:
    - Hệ thống gợi ý (Recommender systems hoặc Recommendation systems) là một dạng của hệ hỗ trợ ra quyết định, cung cấp giải pháp mang tính cá nhân hóa mà không phải trải qua quá trình tìm kiếm phức tạp. Hệ gợi ý học từ người dùng và gợi ý các sản phẩm tốt nhất trong số các sản phẩm phù hợp.
    - Hệ thống gợi ý sử dụng các tri thức về sản phẩm, các tri thức của chuyên gia hay tri thức khai phá học được từ hành vi con người dùng để đưa ra các gợi ý về sản phẩm mà họ thích trong hàng ngàn hàng vạn sản phẩm có trong hệ thống. Các website thương mại điện tử, ví dụ như sách, phim, nhạc, báo...sử dụng hệ thống gợi ý để cung cấp các thông tin giúp cho người sử dụng quyết định sẽ lựa chọn sản phẩm nào. Các sản phẩm được gợi ý dựa trên số lượng sản phẩm đó đã được bán, dựa trên các thông tin cá nhân của người sử dụng, dựa trên sự phân tích hành vi mua hàng trước đó của người sử dụng để đưa ra các dự đoán về hành vi mua hàng trong tương lai của chính khách hàng đó. Các dạng gợi ý bao gồm: gợi ý các sản phẩm tới người tiêu dùng, các thông tin sản phẩm mang tính cá nhân hóa, tổng kết các ý kiến cộng đồng, và cung cấp các chia sẻ, các phê bình, đánh giá mang tính cộng đồng liên quan tới yêu cầu, mục đích của người sử dụng đó.
    
    ##### II. Các phương pháp gợi ý:
    ###### 1. Hệ thống gợi ý dựa theo lọc cộng tác:
    - Hệ thống gợi ý dựa theo lọc cộng tác (Collaborative recommendation systems): là phương pháp gợi ý được triển khai rộng rãi nhất và thành công nhất trong thực tế.
    - Hệ thống theo lọc công tác phân tích và tổng hợp các điểm số đánh giá của các đối tượng, nhận ra sự tương đồng giữa những người sử dụng trên cơ sở các điểm số đánh giá của họ và tạo ra các gợi ý dựa trên sự so sánh này. Hồ sơ (profile) của người sử dụng điển hình trong hệ thống lọc cộng tác bao gồm một vector các đối tượng (item) và các điểm số đánh giá của chúng, với số chiều tăng lên liên tục khi người sử dụng tương tác với hệ thống theo thời gian.
    - Một số hệ thống sử dụng phương pháp chiết khấu dựa trên thời gian (time-based discounting) để tính toán cho yếu tố “trượt” đối với sự quan tâm của người sử dụng. Trong một số trường hợp điểm số đánh giá (rating) có thể là nhị phân (thích/không thích) hoặc các giá trị số thực cho thấy mức độ ưu tiên.
    - Thế mạnh lớn nhất của kỹ thuật gợi ý theo lọc cộng tác là chúng hoàn toàn độc lập với sự biểu diễn của các đối tượng đang được gợi ý, và do đó có thể làm việc tốt với các đối tượng phức tạp như âm thanh và phim. Schafer, Konstan & Riedl (1999) gọi lọc cộng tác là “tương quan giữa người – với – người” (people-to-people correlation).
    
    """)  
    st.image("collaborative.JPG")
    st.write("""
    ###### 2. Hệ thống gợi ý dựa theo nội dung:
    - Hệ thống gợi ý dựa theo nội dung (Content-based recommendation systems): là sự kế thừa và mở rộng của lĩnh vực nghiên cứu lọc thông tin.
    - Trong hệ thống thì các đối tượng được biểu diễn bởi các đặc điểm liên quan tới chúng. Ví dụ, hệ thống gợi ý văn bản như hệ thống lọc tin NewsWeeder sử dụng những từ của các văn bản như các đặc điểm.
    - Một số hệ thống gợi ý dựa trên nội dung học một hồ sơ cá nhân về sở thích của người sử dụng dựa trên các đặc điểm xuất hiện trong chính các đối tượng người sử dụng đã đánh giá (rated). Schafer, Konstan & Riedl gọi gợi ý theo nội dung là “tương quan đối tượng với đối tượng” (item-to-item correlation). Hồ sơ người sử dụng của một hệ thống gợi ý theo nội dung phụ thuộc vào phương pháp học máy được dùng.
    - Cây quyết định (Decision trees), mạng noron (neural nets) và biểu diễn dựa theo vector (vector-based representations) đều có thể được sử dụng để học hồ sơ người dùng. Cũng giống như trong lọc cộng tác, hồ sơ người dùng trong gợi ý dựa theo nội dung là những dữ liệu lâu dài và được cập nhật theo thời gian.
    
    """)
    st.image("content_based.JPG")
    st.write("""
    Ở đây ta sẽ dùng cả collaborative recommendation system để đưa ra gợi ý dựa trên sự tương đồng giữa những người sử dụng, và content-based recommendation system để đưa ra gợi ý dựa trên các đặc điểm liên quan giữa các sản phẩm trên Tiki
    """)
elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('#### 1a. Một vài data với dữ liệu về thông tin, giá cả, thương hiệu v.v. của sản phẩm:')
    # st.dataframe(products[['item_id','name','rating','price','brand']].head(5))
    # st.dataframe(products[['item_id','name','rating','price','brand']].tail(5))
    st.image("somedata_product.jpg")
    st.write('#### 1b. Một vài data với dữ liệu về ID sản phẩm, rating, nội dung đánh giá của khách hàng:')
    # st.dataframe(reviews[['product_id','rating','title','content']].head(5))
    # st.dataframe(reviews[['product_id','rating','title','content']].tail(5))
    st.image("somedata_review.jpg")
    st.write('#### 2. Một số biểu đồ cần xem xét:')
    st.write('###### Biểu đồ giá')
    # fig1 = sns.displot(products, x='price', kind = 'hist')
    # st.pyplot(fig1)
    st.image('price_chart.jpg')
    st.write('Mức giá phổ biến nhất nằm trong khoảng từ 0 - 5.000.000, giá càng cao càng ít sản phẩm')
    
    st.write('###### Biểu đồ rating')
    # fig2 = sns.displot(products, x='rating', kind = 'hist')
    # st.pyplot(fig2)
    st.image('rating_chart.jpg')
    st.write('''
    - Rating 0 và 5 có số lượng tương đương nhau.
    - Rating từ 4.6 đến 5 chiếm tỷ lệ nhiều nhất. Rating dưới 4 chiếm tỷ lệ rất thấp''')
    
    st.write('###### Top 10 thương hiệu bán nhiều nhất')
    # brand = products.groupby('brand')['item_id'].count().sort_values(ascending=False)
    # brand_df = brand[1:11].to_frame().reset_index()
    # brand_df.rename({'item_id':'number_of_items'}, axis=1, inplace=True)
    
    # fig3 = px.bar(brand_df, x='brand', y='number_of_items')
    # fig3.show()
    # st.plotly_chart(fig3)
    st.image('top10_sales.jpg')
    
    st.write('###### Top 10 thương hiệu có mức giá trung bình cao nhất')
    # price_by_brand = products.groupby('brand').mean()['price'].sort_values(ascending = False)
    # price_df = price_by_brand[:10].to_frame().reset_index()
    # fig4 = px.bar(price_df, x='brand', y='price')
    # fig4.show()
    # st.plotly_chart(fig4)
    st.image("top10_prices.jpg")
    
    st.write('###### Top 10 sản phẩm có nhiều lượt đánh giá nhất')
    # top10_product_review = reviews.groupby('product_id').count()['rating'].sort_values(ascending= False)
    # top10_product_review.index = products[products.item_id.isin(top10_product_review.index)]['name'].str[:30]
    # top10_product_df = top10_product_review[:10].to_frame().reset_index()
    # fig5 = px.bar(top10_product_df, x='name', y='rating')
    # fig5.show()
    # st.plotly_chart(fig5)
    st.image('top10_most_reviews.jpg')
    
    st.write('###### Top 10 sản phẩm có rating trung bình cao nhất')
    # top10_rating_review = reviews.groupby('product_id').mean()['rating'].sort_values(ascending= False)
    # top10_rating_review.index = products[products.item_id.isin(top10_rating_review.index)]['name'].str[:30]
    # top10_rating_df = top10_rating_review[:10].to_frame().reset_index()
    # fig6 = px.bar(top10_rating_df, x='name', y='rating')
    # fig6.show()
    # st.plotly_chart(fig6)
    st.image('top10_most_avg_rating.jpg')
    
    st.write('#### 3. Build model')
    st.write('#### 4. Evaluation')
    
    st.write('##### Hiển thị sự tương đồng trên word cloud đối với mô hình content_based với trường hợp item_id = 52333193')
    # wc_text = word_cloud(52333193,5)
    # wc_52333193 = wc_draw(wc_text)
    # plt.axis("off")
    # plt.savefig('wordcloud.jpg')
    # plt.show()
    st.image('wordcloud.jpg')
    
    st.write('##### RMSE và MAE của phương án dùng Collaborative recommenation system')
    # st.table(result_B)
    st.image('rmse.jpg')
    
    st.write('''#### 5. Tổng kết: 
    - Với Content-based filtering: áp dụng khi người dùng search 1 sản phẩm trên website, ta sẽ đề xuất các sản phẩm có tên và mô tả tương tự. Trong bài toán này, nên dùng Content-based solution 1 là Cosine similarity vì phần trăm tương đồng cho ra cao hơn.
    - Với Collaborative filtering: áp dụng khi người dùng đã mua sản phẩm trên website rồi. Sau khi họ đã nhận được hàng và để lại đánh giá, ta có thể gửi email đề xuất các sản phẩm tương đồng cho họ. Trong bài toán này, nên dùng Surprise-BaselineOnly vì có RMSE và MAE tốt hơn các thuật toán khác
    ''')
elif choice == 'Recommendation for User':
    st.subheader("Đề xuất sản phẩm dành cho bạn")
    option_list = contentbased_solution1['product_id'].value_counts().index.values.tolist()
    type = st.sidebar.selectbox("Chọn mô hình:", options=("Content_based","Collaborative"))
    if type == "Content_based":
        type1 = st.selectbox("Download file sẵn có hay nhập mã sản phẩm cụ thể?", options = ("Download file sẵn có",'Chọn sản phẩm cụ thể'))
        if type1 == "Download file sẵn có":
            st.download_button(label="Danh sách 20 sản phẩm đề xuất cho từng sản phẩm", data=csv, file_name='contentbased.csv', mime='csv')
            st.write('Bạn có thể chọn 1 sản phẩm cụ thể trong danh sách để xem đề xuất dành cho mình')
            st.stop()
        else:
            options = st.selectbox('Bạn thích sản phẩm nào?', option_list[:10])
            item_id = options
            
            recommended_item_num = st.slider("Số lượng sản phẩm đề xuất", min_value=2, max_value=10)
            show_recommended_items_btn = st.button("search")
            if recommended_item_num:
                rec_no = recommended_item_num
            if show_recommended_items_btn:
                recommender(item_id,rec_no)
        
         
           
    else:
        options = st.selectbox('Bạn thích sản phẩm nào?', option_list[:10])
        item_id = options
        recommended_item_num = st.slider("Mức rating thấp nhất đề xuất", min_value=2, max_value=5)
        if recommended_item_num:
            rec_no = recommended_item_num
        show_recommended_items_btn = st.button("search")
        if show_recommended_items_btn:
            review_score = reviews[['product_id']]
            review_score['EstimateScore'] = review_score['product_id'].apply(lambda x: collaborative.predict(item_id, x).est)
            review_score = review_score.sort_values(by=['EstimateScore'], ascending=False)
            review_score = review_score.drop_duplicates()
            st.code("Sản phẩm đề xuất cho customer_id "+ str(item_id) + ' là:')
            st.table(review_score.head(rec_no))
            
# rec = recommender(77918557,3)
# st.code("Đề xuất " + str(3) +' sản phẩm tương tự với sản phẩm '+ name(77918557) + ':')
# results = {}
# for idx, row in products.iterrows():    
#     similar_indices = content_based[idx].argsort()[:-22:-1]
#     similar_items = [(content_based[idx][i], products['item_id'][i]) for i in similar_indices]
#     results[row['item_id']] = similar_items[1:]
# recommend = results[77918557][:3]
# for recs in recommend:
#     st.write('*'*10)
#     st.code('Đề xuất : product_id:' + str(recs[1]) + '-' + name(recs[1]) + " với mức tương đồng là " + str(recs[0]))