# NER_task
尝试构建一个做NER任务的项目，分别用pytorch1.6+和TensorFlow2.8+框架来实现；
模型采用biLstm+CRF,Bert,Bert+CRF
标注体系采用BIOSE体系，可以分别使用传统的flat NER和二段式方案来实现
数据集采用GLEU提供的数据集，代码也吸收它里面的思路
