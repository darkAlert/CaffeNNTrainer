#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "classifiertrainer.h"
#include "triplettraineronline.h"
#include "attributepredictortrainer.h"
#include "landmarktrainer.h"
#include "fctrainer.h"
#include "segmenttrainer.h"
#include "cartoontrainer.h"

#define NETWORK_TYPE 2

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
#if NETWORK_TYPE == 0
    std::shared_ptr<ClassifierTrainer> nn_trainer;
#elif NETWORK_TYPE == 1
    std::shared_ptr<TripletTrainerOnline> nn_trainer;
#elif NETWORK_TYPE == 2
    std::shared_ptr<AttributePredictorTrainer> nn_trainer;
#elif NETWORK_TYPE == 3
    std::shared_ptr<LandmarkTrainer> nn_trainer;
#elif NETWORK_TYPE == 4
    std::shared_ptr<FCTrainer> nn_trainer;
#elif NETWORK_TYPE == 5
    std::shared_ptr<SegmentTrainer> nn_trainer;
#elif NETWORK_TYPE == 6
    std::shared_ptr<CartoonTrainer> nn_trainer;
#endif

public slots:
    void Train();
    void Stop();
    void Restore();
    void PauseResume();
};

#endif // MAINWINDOW_H
