#-------------------------------------------------
#
# Project created by QtCreator 2016-07-08T18:44:54
#
#-------------------------------------------------

QT       += core gui
CONFIG   += c++11

QMAKE_CXXFLAGS += -std=c++11
#QMAKE_CXXFLAGS_RELEASE += -O1
#QMAKE_CXXFLAGS_RELEASE += -std=c++11 -g -O2 -ltbb

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = NNTrainer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    helper.cpp \
    triplettraineronline.cpp \
    dataprovideridx.cpp \
    dataprovideridentities.cpp \
    dataprovidersamples.cpp \
    classifiertrainer.cpp \
    attributepredictortrainer.cpp \
    dataproviderattributes.cpp \
    landmarktrainer.cpp \
    fctrainer.cpp \
    segmenttrainer.cpp \
    cartoontrainer.cpp \
    cartoonconstructor.cpp

HEADERS  += mainwindow.h \
    helper.h \
    triplettraineronline.h \
    dataprovideridx.h \
    dataprovideridentities.h \
    dataprovidersamples.h \
    classifiertrainer.h \
    attributepredictortrainer.h \
    dataproviderattributes.h \
    landmarktrainer.h \
    fctrainer.h \
    segmenttrainer.h \
    cartoontrainer.h \
    cartoonconstructor.h

FORMS    += mainwindow.ui

# OpenCV settings for OS X
osx {
    message("* Using settings for OS X.")

    #OpenCV:
    INCLUDEPATH += /usr/local/include
    LIBS += -L/usr/local/lib \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc \
            -lopencv_videoio \
            -lopencv_objdetect \
            -lopencv_cudaobjdetect \
            -lopencv_cudawarping \
            -lopencv_imgcodecs

    #Caffe:
    INCLUDEPATH += /caffe-master/include
    INCLUDEPATH += /usr/local/cuda/include
    INCLUDEPATH +=  /usr/local/Cellar/boost/1.60.0_1/include
    INCLUDEPATH +=  /usr/local/Cellar/glog/0.3.4/include
    INCLUDEPATH +=  /OpenBLAS-0.2.15/build/include
    INCLUDEPATH += /caffe-master/.build_release/
    INCLUDEPATH += /caffe-master/.build_release/src
    INCLUDEPATH += /Developer/NVIDIA/CUDA-7.5/inclue
    LIBS += -L/caffe-master/distribute/lib/ -lcaffe
    LIBS += -L/usr/local/Cellar/glog/0.3.4/lib/ -lglog
    LIBS += -L/usr/local/Cellar/boost/1.60.0_1/lib -lboost_system
    LIBS += -L/Developer/NVIDIA/CUDA-7.5/lib -lcurand
    LIBS += -L/Developer/NVIDIA/CUDA-7.5/lib -lcurand.7.5
}


## settings for Linux
linux {
    message("* Using settings for Linux.")

    QMAKE_LFLAGS += -Wl,--start-group -ldl

    #OpenCV:
    INCLUDEPATH += /usr/local/include/opencv2
    LIBS += -L/usr/local/lib \
            -lopencv_core \
            -lopencv_highgui \
            -lopencv_imgproc \
            -lopencv_videoio \
            -lopencv_objdetect \
            -lopencv_cudaobjdetect \
            -lopencv_cudawarping \
            -lopencv_imgcodecs

    #Caffe:
    INCLUDEPATH += /home/darkalert/Downloads/caffe-master/distribute/include
    INCLUDEPATH += /usr/include/hdf5/serial/
    INCLUDEPATH += /usr/include/glog/
    INCLUDEPATH += /usr/include/boost/
    LIBS += -L/home/darkalert/Downloads/caffe-master/distribute/lib -lcaffe
    LIBS += -L/usr/lib -lglog
    LIBS += -L/usr/lib -lboost_system
    LIBS += -L/usr/local/lib -lprotobuf

    #Cuda:
    INCLUDEPATH += /usr/local/cuda/include/
    LIBS += -L/usr/local/cuda/lib64 -lcurand -lcudnn -lcublas -lcudart
}

DISTFILES += \
    NNTrainer.pro.user \
    NNTrainer.pro.user.78908b7 \
    NNTrainer.pro.user.d7f5022
