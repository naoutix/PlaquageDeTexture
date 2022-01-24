#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>


using namespace cv;
using namespace std;

void printHelp(const string& progName)
{
    cout << "Usage:\n\t " << progName << " <image_file> <K_num_of_clusters> [<image_ground_truth>]" << endl;
}

double kmeansPerso(InputArray data, int K,InputOutputArray bestLabels,TermCriteria criteria,int attempts,int flags,OutputArray centers=noArray())
{
	//Donnes
    int N = data.rows()*data.cols();			// Nombre de points
    int center[K][3];							// Centres
	
	Mat data_1 = data.getMat();					// Valeurs des points

	bestLabels.create(N,1,CV_32S);				// Classes des points
	Mat bestLabels_1  = bestLabels.getMat();

	// **** Random Center ***
	for (int i=0;i<K;i++)
    {
        int randN = rand() % N;
		center[i][0] = data_1.at<float>(randN,0);
		center[i][1] = data_1.at<float>(randN,1);
		center[i][2] = data_1.at<float>(randN,2);
    }
    //cout << "center = "<< endl << " " << center << endl << endl;
	/*
    for (int i=0; i<K; i++){
		for (int j=0; j <3;j++){
			cout << center[i][j] << endl;
		}
    }
	*/

    int maxiter = 0;
    while (maxiter < 1)
    {
        for (int i = 0; i < N; i++)
        {
            double min = 765.0;
            for (int classe = 0; classe < K; classe++)
            {
				// Distance eclidienne entre les points
                double distance = sqrt(
  pow(center[classe][0] - data_1.at<float>(i,2),2) 
+ pow(center[classe][1] - data_1.at<float>(i,1),2) 
+ pow(center[classe][2] - data_1.at<float>(i,2),2));
				//cout << distance << endl;
				// prise du minimum
                if (min > distance)
                {
                    min = distance;
                    bestLabels_1.at<int>(i,0) = classe;
                }
            }
        }
		//cout << "center = "<< endl << " " << bestLabels_1 << endl << endl;
        maxiter++;
    }
    
    return 1.0;
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc !=4)
    {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc ==4) ? string(argv[3]) : string();
    const int k = stoi(argv[2]);

	// Matrices
    Mat image,data,centers,bestLabels,imageTrue;   
    image = imread(argv[1],CV_LOAD_IMAGE_COLOR);

	// *** Conversion des donnees
    image.convertTo(data,CV_32F);
    data = data.reshape(0,image.rows * image.cols);

    //cout << "data = "<< endl << " " << data << endl << endl;

	// *** Kmeans openCV ***
    TermCriteria crit = TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,10,1);
    //double res = kmeans(data,k,bestLabels,crit,4,KMEANS_PP_CENTERS,centers);
	

	// *** Kmeans perso ***
    double mine = kmeansPerso(data,k,bestLabels,crit,4,KMEANS_PP_CENTERS,centers);
    
	// *** Colorisation des classes et creation de l'image ***
	for (int i = 0; i < image.rows*image.cols; i++)
    {
           bestLabels.at<int>(i) = 255-floor(255*bestLabels.at<int>(i)/(k-1));
    }   
    bestLabels= bestLabels.reshape(0,image.rows);
    bestLabels.convertTo(bestLabels,CV_8U);


	// *** Sauvegarde et creation de l'image ***
    imwrite("Segmentation.jpg",bestLabels);
    namedWindow("segmentation",CV_WINDOW_AUTOSIZE);
    imshow("segmentation",bestLabels);


	// *** Comparaison a la reference ***
    if (argc == 4)
    {
        imageTrue =imread(argv[3],CV_LOAD_IMAGE_GRAYSCALE);
        int TP,TN,FP,FN;

        for (int i=0; i< image.rows; i++){
            for (int j=0; j< image.cols; j++){
                if (bestLabels.at<int>(i,j) == imageTrue.at<int>(i,j)){
                    if (bestLabels.at<int>(i,j) == 0){
                        TP++;
                    } else {
                        TN++;
                    }
                } else {
                     if (bestLabels.at<int>(i,j) == 0){
                        FP++;
                    } else {
                        FN++;
                    }
                }
            }
        }       
    }
    waitKey(0);

    return EXIT_SUCCESS;
}
abels 