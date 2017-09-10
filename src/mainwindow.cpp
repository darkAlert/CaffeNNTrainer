#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    //GUI:
    ui->setupUi(this);
    connect(ui->pushButton, SIGNAL(released()), this, SLOT(Train()));
    connect(ui->pushButton_2, SIGNAL(released()), this, SLOT(Stop()));
    connect(ui->pushButton_3, SIGNAL(released()), this, SLOT(Restore()));
    connect(ui->pushButton_4, SIGNAL(released()), this, SLOT(PauseResume()));

    //Init:
#if NETWORK_TYPE == 0
    nn_trainer = std::make_shared<ClassifierTrainer>();
#elif NETWORK_TYPE == 1
    nn_trainer = std::make_shared<TripletTrainerOnline>();
#elif NETWORK_TYPE == 2
    nn_trainer = std::make_shared<AttributePredictorTrainer>();
#elif NETWORK_TYPE == 3
    nn_trainer = std::make_shared<LandmarkTrainer>();
#elif NETWORK_TYPE == 4
    nn_trainer = std::make_shared<FCTrainer>();
#elif NETWORK_TYPE == 5
    nn_trainer = std::make_shared<SegmentTrainer>();
#elif NETWORK_TYPE == 6
    nn_trainer = std::make_shared<CartoonTrainer>();
#endif
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::Train()
{ 
#if NETWORK_TYPE == 0
    /// NORMAL AND SIAMESE TRAINING ///
    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/facial_component_144x144/eye/csv.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/classifier_eyes/solver.prototxt";
    std::string path_to_testnet = "";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver, path_to_testnet);

#elif NETWORK_TYPE == 1
    /// Triplet Online ///
    std::string path_to_train_data = "/ssd490/ms_and_mega_all.txt";
    std::string path_to_solver = "/home/vitvit/snap/tresnet256maxout/solver.prototxt";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);

#elif NETWORK_TYPE == 2
    /// Attribute Predictor ///
    /*
    //Face attributes:
    std::string path_to_train_data = "/home/darkalert/Desktop/MirrorJob/Datasets/Processed/attributes2/ms200k_200x150_train.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/attributes_net_1/qsolver.prototxt";
    */
     /*
    //Facial components:
    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/FFUD_clean_components/hair_back_and_hair_2814.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/classifier_hairs/solver.prototxt";
    */

    //Facial components:
    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/FFUD_clean_components/hair_back_and_hair_2814.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/classifier_hairs/solver.prototxt";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);

#elif NETWORK_TYPE == 3
    /// Landmark Trainer ///
//    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_aligned/51p_landmarks_9set.csv";
    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_aligned/68p_landmarks_9set.csv";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/68p_resnet1/qsolver.prototxt";
//    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_rect/rect_9set.csv";
//    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/rect_resnet1/qsolver.prototxt";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);
#elif NETWORK_TYPE == 4
    /// FullyConnected Trainer ///
    std::string path_to_data = "/home/darkalert/Desktop/MirrorJob/Datasets/Processed/landmarks_transform/style1_69as68/csv.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/qsolver.prototxt";

    qDebug() << "Setting data...";
    nn_trainer->read_raw_samples(path_to_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);
#elif NETWORK_TYPE == 5
    /// Segmentation Trainer ///
    std::string path_to_data = "/home/darkalert/MirrorJob/Datasets/Processed/segmentation/full_z1p4_160x160/csv.txt";
//    std::string path_to_data = "/home/darkalert/MirrorJob/Datasets/Processed/segmentation/hair_120x160/csv.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/segment_resnet1/qsolver.prototxt";



    qDebug() << "Setting data...";
    nn_trainer->read_raw_samples(path_to_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);
#elif NETWORK_TYPE == 6
    /// Cartoon Trainer ///
    std::string path_to_data = "/home/darkalert/MirrorJob/Datasets/Processed/landmarks_transform/style1_69as68/csv_exclude_s2.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/qsolver.prototxt";

    qDebug() << "Setting data...";
    nn_trainer->read_raw_samples(path_to_data);
    qDebug() << "Run training...";
    nn_trainer->train(path_to_solver);
#endif
}

void MainWindow::Stop()
{
    nn_trainer->stop();
    qDebug() << "Training has been stopped.";
}

void MainWindow::Restore()
{
#if NETWORK_TYPE == 0 || NETWORK_TYPE == 1
    /// NORMAL AND SIAMESE TRAINING ///
    std::string path_to_train_data = "/ssd490/ms10575.txt";
    std::string path_to_solver = "/home/vitvit/snap/resnet50/qsolver.prototxt";
    std::string path_to_testnet = "/home/vitvit/snap/resnet50/qtest.prototxt";
    std::string path_to_solverstate = "/home/vitvit/snap/resnet50/snap/_iter_20000.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);

    qDebug() << "Restoring...";
    nn_trainer->restore(path_to_solver, path_to_solverstate, path_to_testnet);

#elif NETWORK_TYPE == 1
    /// Triplet Online ///
    std::string path_to_train_data = "/ssd490/MS_Celeb/msclean_train.txt";
    std::string path_to_solver = "/home/vitvit/snap/tresnet256maxout/solver.prototxt";
    std::string path_to_solverstate = "/home/vitvit/snap/tresnet256maxout/snaps/_iter_19000.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Restoring...";
    nn_trainer->restore(path_to_solver, path_to_solverstate);

#elif NETWORK_TYPE == 2
    /// Attribute Predictor ///
    //Face attributes:
    /*
    std::string path_to_train_data = "/home/darkalert/Desktop/MirrorJob/Datasets/Processed/attributes/ms200k_200x150_train.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/attributes_net_1/qsolver.prototxt";
    std::string path_to_solverstate = "/home/darkalert/MirrorJob/Snaps/attributes_net_1/snaps/v7_iter_34079.solverstate";
    */

    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/FFUD_components/oval_rgb_train_l12_22656.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/classifier_ovals/solver.prototxt";
    std::string path_to_solverstate = "/home/darkalert/MirrorJob/Snaps/classifier_ovals/snaps/oval_xs_rgb_l12_iter_70000.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);

    qDebug() << "Restoring...";
    nn_trainer->restore(path_to_solver, path_to_solverstate);

#elif NETWORK_TYPE == 3
    /// Landmark Trainer ///
//    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_aligned/51p_landmarks_9set.csv";
    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_aligned/68p_landmarks_9set.csv";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/68p_resnet1/qsolver.prototxt";
    std::string path_to_solverstate = "/home/darkalert/MirrorJob/Snaps/68p_resnet1/best/v1_iter_35000.caffemodel";

//    std::string path_to_train_data = "/home/darkalert/MirrorJob/Datasets/Processed/ibug2013_rect/rect_9set.csv";
//    std::string path_to_solver = "/home/darkalert/Desktop/MirrorJob/Snaps/rect_resnet1/qsolver.prototxt";
//    std::string path_to_solverstate = "/home/darkalert/Desktop/MirrorJob/Snaps/rect_resnet1/snaps/fc_rotate0_iter_67520.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->openTrainData(path_to_train_data);
    qDebug() << "Restoring...";
    nn_trainer->restore(path_to_solver, path_to_solverstate);

#elif NETWORK_TYPE == 4
    /// FullyConnected Trainer ///
    std::string path_to_data = "/home/darkalert/Desktop/MirrorJob/Datasets/Processed/landmarks_transform/style_t/csv.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/qsolver.prototxt";
    std::string path_to_solverstate = "/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/snap/v1_iter_50000.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->read_raw_samples(path_to_data);
    qDebug() << "Run training...";
    nn_trainer->restore(path_to_solver, path_to_solverstate);
#elif NETWORK_TYPE == 5
    /// Segmentation Trainer ///
    std::string path_to_data = "/home/darkalert/MirrorJob/Datasets/Processed/segmentation/segments2/csv.txt";
    std::string path_to_solver = "/home/darkalert/MirrorJob/Snaps/segment_resnet1/qsolver.prototxt";
    std::string path_to_solverstate = "/home/darkalert/MirrorJob/Snaps/landmarks_transform_1/snap/v1_iter_50000.solverstate";

    qDebug() << "Setting data...";
    nn_trainer->read_raw_samples(path_to_data);
    qDebug() << "Run training...";
    nn_trainer->restore(path_to_solver, path_to_solverstate);
#endif
}

void MainWindow::PauseResume()
{
    //Pause:
    if (ui->pushButton_4->text().compare(QString("Pause")) == 0) {
        nn_trainer->pause();
        ui->pushButton_4->setText(QString("Resume"));
    }
    //Resume:
    else {
        nn_trainer->resume();
        ui->pushButton_4->setText(QString("Pause"));
    }
}
