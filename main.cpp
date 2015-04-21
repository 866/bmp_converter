#include <iostream>
#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
#include <lmdb.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "caffe/proto/caffe.pb.h"

using namespace std;
using namespace boost::filesystem;
using namespace boost::algorithm;
using namespace caffe;
using namespace cv;

using boost::lexical_cast;
typedef vector< string > split_vector_type;

enum LABEL_SET {DIGITS, CAP_LETTERS, SMALL_LETTERS};

int findBlobParams(Mat& img, Rect& bounds, Point2f& centroid) //finds blob's bounds and centroid
                                                                      //returns number of blobs points
{
    uchar* img_ptr;
    int blob_points = 0;
    bounds = Rect(img.size().width, img.size().height, 0, 0);
    centroid = Point2f(0, 0);

    for(int y = 0; y < img.size().height; ++y)
    {
        img_ptr = img.ptr(y);
        for(int x = 0; x < img.size().width; ++x)
        {
            if (img_ptr[x] > 0)
            {
                blob_points++;
                if (bounds.x > x)
                    bounds.x = x;
                if (bounds.width < x)
                    bounds.width = x;
                if (bounds.y > y)
                    bounds.y = y;
                if (bounds.height < y)
                    bounds.height = y;
                centroid += Point2f(x,y);
            }
        }
    }

    bounds.height -= (bounds.y-1);
    bounds.width -= (bounds.x-1);
    centroid = (centroid/blob_points)-Point2f(bounds.x, bounds.y);

    //    cout << "bounds.x = " << bounds.x << endl
    //         << "bounds.y = " << bounds.y << endl
    //         << "bounds.height = " << bounds.height << endl
    //         << "bounds.width = " << bounds.width << endl
    //         << "Found " << blob_points << " points" << endl
    //         << "Center of masses " << centroid;
}

char getClassNumbers(char label) // returns the class of the given character with respect to digits set
                                // based on ASCII characters
                                // 11 - unknown class
{
    char res = static_cast<int>(label)-48;
    if ((res > 10) || (res < 0))
            res = 11;
    return res;
}

char getClassCapLetters(char label) // returns the class of the given character with respect to capital letters set
                                   // based on ASCII characters
                                   // 26 - unknown class
{
    char res = static_cast<int>(label)-65;
    if ((res > 25) || (res < 0))
            res = 26;
    return res;
}

char getClassSmallLetters(char label) // returns the class of the given character with respect to small letters set
                                   // based on ASCII characters
                                   // 26 - uknkown class
{
    char res = static_cast<int>(label)-97;
    if ((res > 25) || (res < 0))
            res = 26;
    return res;
}

void convertImageToLeNet(Mat& img) // opens bitmap file and returns byte array with the pixels of transformed image
{
    Rect bounds;
    Point2f cm;

    bitwise_not(img, img); // color inversion for LeNet

    findBlobParams(img, bounds, cm);//finds certain character position

    int max_side = static_cast<int>((max(bounds.width, bounds.height))*1.4); // image should have sides equal to maximum side of character's frame

    Mat characterImage = Mat::zeros(max_side, max_side, img.type()); // here we will store image that is compatible with LeNet
    Point2i disp = (Point2f(0.5*max_side, 0.5*max_side)-cm);// displacement of cantroid

    cv::Mat destinationROI = characterImage( Rect(disp.x, disp.y, bounds.width, bounds.height) ); // select region of character
    cv::Mat sourceROI =  img( bounds ); // select region of character
    sourceROI.copyTo(destinationROI); // copy character to characterImage

    resize(characterImage, img, Size(28,28)); // resize to desired size
}

short getLabel(string path, LABEL_SET type)
{
    split_vector_type SplitVec;
    split( SplitVec, path, is_any_of("_"), token_compress_on );
    char clabel = SplitVec[SplitVec.size()-2].c_str()[0]; // get a label of image
    switch (type)
    {
        case DIGITS:
            return getClassNumbers(clabel);
        case CAP_LETTERS:
            return getClassCapLetters(clabel);
        case SMALL_LETTERS:
            return getClassSmallLetters(clabel);
    }
}

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    cout << "Usage: bmp_converter <path> <db>\n";
    return 1;
  }

  path p (argv[1]);


if (exists(p))    // does p actually exist?
    {

      if (is_regular_file(p))        // is p a regular file?
        cout << "<path> should be directory not a file" << '\n';

      else if (is_directory(p))      // is p a directory?
      {

        // lmdb
        MDB_env *mdb_env;
        MDB_dbi mdb_dbi;
        MDB_val mdb_key, mdb_data;
        MDB_txn *mdb_txn;

        // Caffe neural network blob
        Datum datum;
        datum.set_channels(1);
        datum.set_height(28);
        datum.set_width(28);

        //additional vars
        typedef vector<path> vec;             // store paths,
        vec v;                                // so we can sort them later
        char label;                           // label
        const int kMaxKeyLength = 10;
        char key_cstr[kMaxKeyLength];
        char *pixels, *db_path = argv[2];
        string value;
        int item_id = 0;

        LOG(INFO) << "Opening lmdb " << db_path;
        CHECK_EQ(mkdir(db_path, 0744), 0)
            << "mkdir " << db_path << "failed";
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";

        copy(directory_iterator(p), directory_iterator(), back_inserter(v));
        for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
        {

            string name = it->string();

            if ((is_regular_file(*it)) &&
                    (name.size()>3) &&
                    (name.compare(name.size()-3, 3, "bmp") == 0)) //find bmp file
            {
                Mat img(imread(name, CV_LOAD_IMAGE_GRAYSCALE)); // image from name
                convertImageToLeNet(img);
                pixels = reinterpret_cast<char*> (img.ptr());
                label = getLabel(name, DIGITS);

                datum.set_data(pixels, 28*28);
                datum.set_label(label);
                snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
                datum.SerializeToString(&value);
                string keystr(key_cstr);

                mdb_data.mv_size = value.size();
                mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
                mdb_key.mv_size = keystr.size();
                mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
                CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
                    << "mdb_put failed";

                item_id++;
            }
        }
        //close db
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);

        cout << item_id << " items have been processed and stored to the database.";
      }
      else
        cout << p << " exists, but is neither a regular file nor a directory\n";
    }


  return 0;
}
