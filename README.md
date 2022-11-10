# IMDB Sentiment Classification
Nama: Rifky Bujana Bisri

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Masalah | Dalam industri film pada umumnya, kostumer dapat menentukan film apa yang ingin ditonton berdasarkan penilaian dari film-film yang ada. Namun terkadang penilaian tersebut dapat bersifat bias karena dengan kemudahan memberikan rating terhadap suatu film seseorang dapat mempermainkan rating dari suatu film dengan mudah. Oleh karena itu penilaian berdasarkan tulisan atau komentar terhadap film dapat bisa lebih dipercaya namun untuk menentukan suatu penilaian komentar tersebut baik atau buruk perlu resource yang sangat banyak jika dilakukan secara tradisional. |
| Solusi machine learning | Pemanfaatan machine learning untuk melakukan penilaian terhadap penulisan komentar terhadap suatu film akan dapat membantu para customer untuk mendapatkan hasil penilaian dari suatu film yang lebih dapat dipercaya. |
| Metode pengolahan | Data berupa teks dilakukan proses pembersihan dengan pengecilan huruf dan pembuangan stopwords. |
| Arsitektur model | Model yang digunakan merupakan model neural networks sederhana yang ditambahkan berupa text vectorization embeddings, sebuah 1D average pooling layer, dan 1 dense layer dengan 32 unit yang akan diteruskan pada sebuah dense layer dengan 1 unit yang berperan sebagai output layer yang dilengkapi dengan sigmoid activation untuk melakukan binary classification. |
| Metrik evaluasi | Task ini merupakan sebuah task binary classification, sehingga saya menggunakan binary-crossentropy loss function serta accuracy metrik untuk melakukan evaluasi dari model. |
| Performa model | Pada saat pelatihan model merupakan goodfit karena nilai metrik dari *training dataset* dan *test dataset* selaras walau model berhenti pada *epochs* ke-13 dengan nilai *validation accuracy* sebesar **70%**. Nilai akurasi tersebut cukup baik jika kita membandingkan dengan arsitektur dari model yang sangat sederhana. Karena arsitekturnya yang sederhana, model dapat memprediksi dengan sangat cepat walau hanya menggunakan CPU. |
| Opsi deployment | Model dideploy menggunakan tensorflow serving pada cloud *heroku*. |
| Web app | https://imdb-prediction.herokuapp.com/ |
| Monitoring | Disini saya menggunakan prometheus untuk melakukan monitoring terhadap inference ditambahkan dengan grafana untuk membangun dashboard agar mempermudah melakukan monitoring terhadap inference. Pada dashboard ini, data yang masuk dan data yang dikirim oleh inference dapat dimonitor dengan mudah. |
