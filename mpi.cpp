#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/matx.hpp>
#include "opencv2/core/core_c.h"

#include <iostream>
#include <iomanip>

#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#include <cmath>

#define SOURCEVID "./prueba2.mp4"

#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */
#define TOTAL_FRAMES 1159

using namespace std;
using namespace cv;

// void detectAndDisplay(Mat frame);
void detectAndDisplay(Mat frame, int frame_width, int frame_height, double fps);
void box_blur(Mat3i frame, Mat outFrame, int r);
Mat3i sumed_table(Mat3i frame);

CascadeClassifier face_cascade;
VideoCapture cap;
VideoWriter video;

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    // Para medir tiempo
    struct timeval ti, tf;
    double tiempo;

    gettimeofday(&ti, NULL); // Instante inicial

    String face_cascade_name = "./haarcascade_frontalface_alt.xml";

    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    // Create a VideoCapture object and use camera to capture the video
    cap.open(SOURCEVID);

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
        error;  /*error*/
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

        // Default resolutions of the frame are obtained.The default resolutions are system dependent.
        int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(CAP_PROP_FPS);

        if (taskid == MASTER)
        {
            // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
            video = VideoWriter("video_out.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(frame_width, frame_height));
        }

        double TFrames = cap.get(CAP_PROP_FRAME_COUNT);

        int frames_per_worker = (int)TFrames / numtasks;
        int start = frames_per_worker * taskid;
        int end = start + (frames_per_worker - 1);

        // if (taskid == (numtasks - 1) && taskid > 1)
        // {
        //     end = TFrames;
        // }

        cout << "start " << start << " end " << end << endl;

        vector<Mat> frames((end - start + 1) * sizeof(Mat));

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
            detectAndDisplay(frame, frame_width, frame_height, fps);
            frames[i] = frame.clone();
        }

        // Instante final for time
        gettimeofday(&tf, NULL);
        tiempo = (tf.tv_sec - ti.tv_sec) * 1000 + (tf.tv_usec - ti.tv_usec) / 1000.0;
        cout << "El Tiempo que tardo en obtener los frames fue " << tiempo / 60000 << endl;

        gettimeofday(&ti, NULL); // Instante inicial
        /**************************** master task ************************************/

        // MPI_Barrier(MPI_COMM_WORLD);

        int heigth = frames[0].rows;
        int width = frames[0].cols;

        /* Measure start time */
        double start_time = MPI_Wtime();

        if (taskid == MASTER)
        {
            printf("mpi_mm has started with %d tasks.\n", numtasks);
            for (int i = 0; i < frames_per_worker * numtasks; i++)
            {
                // cout << i<<endl;
                if (start <= i && i <= end)
                {
                    video.write(frames[i]);
                    // cout << "i " << i << endl;
                }
                else
                {
                    int size_vector = heigth * width * 3;
                    int node = (int)i / (end - start + 1);

                    uchar *frame_to_vector = (uchar *)malloc(size_vector * sizeof(uchar));
                    MPI_Recv(frame_to_vector, size_vector, MPI_UNSIGNED_CHAR, node, 0, MPI_COMM_WORLD, &status);
                    Mat frame_rec = Mat(heigth, width, CV_8UC3, frame_to_vector);
                    video.write(frame_rec);

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
        /* Measure start time */
        double end_time = MPI_Wtime();

        cout << end_time - start_time << endl;
    }

    // Instante final for time
    gettimeofday(&tf, NULL);
    tiempo = (tf.tv_sec - ti.tv_sec) * 1000 + (tf.tv_usec - ti.tv_usec) / 1000.0;
    cout << "El Tiempo que tardo en blur fue " << tiempo / 60000 << endl;

    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

//-- funtion for blur frame
void box_blur(Mat3i frame, Mat outFrame, int r)
{
    Mat3i table, newFrame;
    frame.copyTo(newFrame);

    //-- Size of frame
    int width = frame.cols;
    int height = frame.rows;

    int area = (2 * r + 1) * (2 * r + 1);

    //-- Sum table
    table = sumed_table(frame);

    Vec<int, 3> areaVector(area, area, area);

    //-- Bucle for aply blur in frame
    for (int x = r + 1; x <= width - r - 1; x++)
    {

        for (int y = r + 1; y <= height - r - 1; y++)
        {

            Vec<int, 3> sum_pixels = (table[y + r][x + r] - table[y + r][x - r - 1] - table[y - r - 1][x + r] + table[y - r - 1][x - r - 1]);
            divide(sum_pixels, areaVector, newFrame[x][y]);
        }
    }
    newFrame.convertTo(outFrame, CV_8U);
}

//-- Function for calculated sum table for frame
Mat3i sumed_table(Mat3i frame)
{
    int width = frame.cols;
    int height = frame.rows;

    Mat3i table = Mat::zeros(height, width, CV_8UC3);

    table[0][0] = frame[0][0];

    //-- Bucle for x-axis[0] for sum table
    for (int x = 1; x < width; x++)
    {
        table[0][x] = frame[x][0] + table[0][x - 1];
    }

    //-- Bucle for y-axis[0] for sum table
    for (int y = 1; y < height; y++)
    {
        table[y][0] = frame[0][y] + table[y - 1][0];
    }

    //-- Bucle for x-axis and y-axis for sum table [x][y]
    for (int x = 1; x <= width - 1; x++)
    {
        for (int y = 1; y <= height - 1; y++)
        {
            table[y][x] = ((frame[x][y] + table[y - 1][x]) + (table[y][x - 1])) - table[y - 1][x - 1];
        }
    }
    return table;
}

//-- Function for detect and blur
void detectAndDisplay(Mat frame, int frame_width, int frame_height, double fps)
{
    //-- Frame for detection
    Mat frame_gray;
    //-- Frame for blur output
    Mat frame_detected;

    frame.copyTo(frame_detected);

    //-- Functions for GrayScale
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        //-- Points of reference for zone detection face
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        Point puntoInicial(faces[i].x + 0, faces[i].y + 0);
        Point puntofinal(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        int width = faces[i].width;
        int height = faces[i].height;
        Size kSize = Size((int)faces[i].width, (int)faces[i].height);
        Rect rect(center.x - width / 2, center.y - height / 2.5, width, height);
        Mat faceROI = frame_gray(faces[i]);

        //-- Function for blur
        box_blur(frame_detected(rect), frame_detected(rect), 10);
    }

    frame_detected.convertTo(frame, CV_8UC3);
    // return frame_detected;
}
