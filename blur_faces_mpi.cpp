#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <mpi.h>

using namespace cv;
using namespace std;

#define THREADS 8
#define PAD 8
#define MAXTASKS 32
#define MASTER 0 /* taskid of first task */

CascadeClassifier face_cascade;
VideoCapture cap;
VideoWriter output;

void faceblur(Mat frame);

int main(int argc, char *argv[])
{
    char path[100] = "./";
    char path2[100] = "./";
    strcat(path, argv[1]);
    strcat(path2, argv[2]);
    int delta = 0;
    int error;

    face_cascade.load("./haarcascade_frontalface_alt.xml");

    cap.open(path);
    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream" << endl;
        return -1;
    }
    else
    {
        int numtasks, /* number of tasks in partition */
            taskid,   /* a task identifier */
            error;    /*error*/
        MPI_Status status;

        // Initialising MPI
        if ((error = MPI_Init(&argc, &argv)) != MPI_SUCCESS)
        {
            fprintf(stderr, "Failed to initialize MPI\n");
            return error;
        }
        // Initialising MPI Communicator Rank
        if ((error = MPI_Comm_rank(MPI_COMM_WORLD, &taskid)) != MPI_SUCCESS)
        {
            fprintf(stderr, "Failed to initialise MPI communicator group\n");
            return error;
        }
        // Initialising MPI Communicator Size
        if ((error = MPI_Comm_size(MPI_COMM_WORLD, &numtasks)) != MPI_SUCCESS)
        {
            fprintf(stderr, "Failed to initialise MPI communicator size\n");
            return error;
        }

        int frame_width = (int)(cap.get(3));
        int frame_height = (int)(cap.get(4));
        Size frame_size(frame_width, frame_height);
        int fps = cap.get(CAP_PROP_FPS);
        // int fps = 20;
        int totalFrames = (int)cap.get(7);

        if (taskid == MASTER)
        {
            // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
            output = VideoWriter(path2, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame_size);
        }

        cout << "Numero de cuadros" << totalFrames << endl;
        cout << "Ancho de cuadro" << frame_width << endl;
        cout << "Alto de cuadro" << frame_height << endl;

        int cont = 0;
        long int cont2 = 0;
        clock_t start_t = clock();

        int frames_per_worker = (int)totalFrames / numtasks;
        int start = frames_per_worker * taskid;
        int end = start + (frames_per_worker - 1);

        cout << "TF " << totalFrames << endl;
        cout << "FPW " << frames_per_worker << endl;
        cout << "start " << start << " end " << end << endl;

        vector<Mat> frames((totalFrames) * sizeof(Mat));

        cap.set(CAP_PROP_POS_FRAMES, (start)*1.0);

        Mat frame;
        for (int i = 0; i < end - start + 1; i++)
        {
            cap >> frame;

            if (frame.empty())
            {
                cout << "No Frame\n";
                break;
            }
            //-- Return a frame whit blur
            faceblur(frame);
            frames[i] = frame.clone();
        }

        int heigth = frames[0].rows;
        int width = frames[0].cols;

        if (taskid == MASTER)
        {
            printf("mpi_mm has started with %d tasks.\n", numtasks);
            for (int i = 0; i < frames_per_worker * numtasks; i++)
            {
                // cout << i<<endl;
                if (start <= i && i <= end)
                {
                    output.write(frames[i]);
                    // cout << "i " << i << endl;
                }
                else
                {
                    int size_vector = heigth * width * 3;
                    int node = (int)i / (end - start + 1);

                    uchar *frame_to_vector = (uchar *)malloc(size_vector * sizeof(uchar));
                    MPI_Recv(frame_to_vector, size_vector, MPI_UNSIGNED_CHAR, node, 0, MPI_COMM_WORLD, &status);
                    Mat frame_rec = Mat(heigth, width, CV_8UC3, frame_to_vector);
                    output.write(frame_rec);

                    // Free Memory
                    free(frame_to_vector);
                }
            }
        }
        else
        {
            for (int i = 0; i < end - start + 1; i++)
            {
                int size_vector = heigth * width * 3;
                uchar *frame_to_vector = frames[i].data;
                MPI_Send(frame_to_vector, size_vector, MPI_UNSIGNED_CHAR, MASTER, 0, MPI_COMM_WORLD);
            }
        }

        cout << "TIempo total: " << float(start_t) / CLOCKS_PER_SEC << endl;
    }
    MPI_Finalize();
}

void faceblur(Mat frame)
{
    vector<Rect> faces;

    int id = omp_get_thread_num();
    face_cascade.detectMultiScale(frame, faces, 1.1, 3, 0);
    int x;
    int y;
    int height;
    int width;

    for (size_t i = 0; i < faces.size(); i++)
    {
        x = faces[i].x;
        y = faces[i].y;
        height = faces[i].height;
        width = faces[i].width;
        int nb = omp_get_thread_num();
        // Range rows(x, (x + height) + (nb * PAD));
        // Range cols(y, (y + width) + (nb * PAD));

#pragma omp num_threads(THREADS)
        for (int j = THREADS * PAD; j < height + (nb * PAD); j++)
        {
            Range rows(x, (x + height));
            for (int k = THREADS * PAD; k < width + (nb * PAD); k++)
            {
                Range cols(y, (y + width));
                blur(frame(cols, rows), frame(cols, rows), Size(45, 45), Point(-1, -1), BORDER_REFLECT);
            }
        }
    }
    faces.clear();
}
