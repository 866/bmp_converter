#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdio>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <boost/thread.hpp>
#include <lmdb.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include "caffe/proto/caffe.pb.h"

#define IMAGE_SIZE 28
#define SCALE_FACTOR 1.5
#define MAXIMUM_NUMBER_OF_THREADS 10
#define DISPLAY_PERIOD 3000


using namespace std;
using namespace boost::filesystem;
using namespace boost::algorithm;
using namespace boost::this_thread;
using namespace caffe;
using namespace cv;

using boost::mutex;
using boost::thread;
using boost::shared_ptr;

typedef vector< string > split_vector_type;
typedef vector< path > vec;             // store paths

enum LABEL_SET {DIGITS, CAP_LETTERS, SMALL_LETTERS};

struct LMDB_DESCRIPTOR //principle variables for lmdb
{
    // shared variables
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;
    uint files_number;
    mutex mtx_;
    int getFileIndex() const {return file_idx;}

    // methods
    LMDB_DESCRIPTOR() : items_index(0), files_number(0),
                        file_idx(0) {}
    void lock() { mtx_.lock(); }
    void unlock() { mtx_.unlock(); }
    int increaseItemsCounter()
    {
	// lock mutex when we need to increase counter
        mtx_.lock();
        int ret = items_index;
        items_index++;
        mtx_.unlock();
        return ret;
    }
    void increaseCurrentFileIndex()
    {
        mtx_.lock();
        file_idx ++;
        mtx_.unlock();
    }

private:
    int items_index, file_idx; //number of items

};

LABEL_SET TARGET_SET;

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
    centroid = (centroid*(1./static_cast<float>(blob_points)))-Point2f(bounds.x, bounds.y);

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
            res = -1;
    return res;
}

char getClassCapLetters(char label) // returns the class of the given character with respect to capital letters set
                                   // based on ASCII characters
                                   // 26 - unknown class
{
    char res = static_cast<int>(label)-65;
    if ((res > 25) || (res < 0))
            res = -1;
    return res;
}

char getClassSmallLetters(char label) // returns the class of the given character with respect to small letters set
                                   // based on ASCII characters
                                   // 26 - uknkown class
{
    char res = static_cast<int>(label)-97;
    if ((res > 25) || (res < 0))
            res = -1;
    return res;
}

int argToClass(char c)
{

    switch(c)
    {
        case 'd':  // digits
        case 'D':
            return DIGITS;

        case 's':  // small letters
        case 'S':
            return SMALL_LETTERS;

        case 'c':
        case 'C':
            return CAP_LETTERS;

    }

    return -1;
}

string classToString(int c)
{

    switch(c)
    {
        case DIGITS:
            return "DIGITS";

        case SMALL_LETTERS:
            return "SMALL_LETTERS";

        case CAP_LETTERS:
            return "CAP_LETTERS";
    }

    return "INCORRECT";
}

int convertImageToLeNet(Mat& img) // opens bitmap file and returns byte array with the pixels of transformed image
{
    Rect bounds;
    Point2f cm;

    bitwise_not(img, img); // color inversion for LeNet

    findBlobParams(img, bounds, cm);  //finds certain character position

    int max_side = static_cast<int>((max(bounds.width, bounds.height))*SCALE_FACTOR); // image should have sides equal to maximum side of character's frame

    try
    {
        Mat characterImage = Mat::zeros(max_side, max_side, img.type()); // here we will store image that is compatible with LeNet
        Point2i disp = (Point2f(0.5*max_side, 0.5*max_side)-cm);// displacement of centroid

        int yieldX = characterImage.size().width - (disp.x + bounds.width); // cut the boundaries,
        int yieldY = characterImage.size().height - (disp.y + bounds.height); // if cut image does not fit destination image
        if (yieldX < 0)
            bounds.width += yieldX;
        if (yieldY < 0)
            bounds.height += yieldY;

        if (disp.x < 0)
            disp.x = 0;
        if (disp.y < 0)
            disp.y = 0;

        cv::Mat destinationROI = characterImage( Rect(disp.x, disp.y, bounds.width, bounds.height) ); // select region of character
        cv::Mat sourceROI =  img( bounds ); // select region of character
        sourceROI.copyTo(destinationROI); // copy character to characterImage
        resize(characterImage, img, Size(IMAGE_SIZE,IMAGE_SIZE)); // resize to desired size
    }
    catch (cv::Exception)
    {
        return -1;
    }


    return 0;
}

char getLabel(string path, LABEL_SET type) // returns class of the label
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

void safeStoreToDB(LMDB_DESCRIPTOR* lmdb, string& value, string& keystr)
{
    lmdb->lock(); //Create thread-safe access to lmdb
    lmdb->mdb_data.mv_size = value.size();
    lmdb->mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    lmdb->mdb_key.mv_size = keystr.size();
    lmdb->mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(lmdb->mdb_txn, lmdb->mdb_dbi, &lmdb->mdb_key, &lmdb->mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";
    lmdb->unlock(); //Unlock access to lmdb
}

void lmdbThread(LMDB_DESCRIPTOR* lmdb, const path* p)///LMDB_DESCRIPTOR &lmdb, path p) //separate thread for working with certain folder
{
    // Caffe neural network blob
    Datum datum;
    datum.set_channels(1);
    datum.set_height(IMAGE_SIZE);
    datum.set_width(IMAGE_SIZE);

    // Additional variables
    char *pixels, label;                  // image and class
    const int kMaxKeyLength = 10;         // maximum number of character in key
    char key_cstr[kMaxKeyLength];
    int item_no = 0;
    string value;
    vec dir_elements;

    copy(directory_iterator(*p), directory_iterator(), back_inserter(dir_elements));

    for (vec::const_iterator it (dir_elements.begin()); it != dir_elements.end(); ++it)
    {
        string name = it->string();

        if ((is_regular_file(*it)))
        {
            lmdb->increaseCurrentFileIndex();
            if ((name.size()>3) &&
                    (name.compare(name.size()-3, 3, "bmp") == 0)) //find bmp file
            {
                label = getLabel(name, TARGET_SET);

                if (label == -1)
                {
                    //LOG(INFO) << "not passed"<< std::endl;
                    continue;
                }
//                LOG(INFO) << "passed"<< std::endl;

                Mat img(imread(name, CV_LOAD_IMAGE_GRAYSCALE)); // image from name
                if (convertImageToLeNet(img) == -1) //replace img by the LeNet img
                {
                    LOG(INFO) << name << " abnormal" << std::endl;
                    continue;
                }
                pixels = reinterpret_cast<char*> (img.ptr());

                item_no = lmdb->increaseItemsCounter();
                datum.set_data(pixels, IMAGE_SIZE*IMAGE_SIZE);
                datum.set_label(label);
                snprintf(key_cstr, kMaxKeyLength, "%08d", item_no);
                datum.SerializeToString(&value);
                string keystr(key_cstr);
                safeStoreToDB(lmdb, value, keystr);

                if (item_no%DISPLAY_PERIOD == 0)
                    LOG(INFO) << lmdb->getFileIndex() << '('<< item_no << " items) out of "<< lmdb->files_number << '('
                              << static_cast<float>(lmdb->getFileIndex())/lmdb->files_number*100 << "%) files have been processed." << std::endl;
            }
        }
    }

//    LOG(INFO) << "End thread in " << *p << " folder";

}

ulong getFilesNumber(const path& p)
{
    vec dir_elements;
    ulong elements_number = 0;
    LOG(INFO) << "Counting files within folders..."<< std::endl;
    copy(directory_iterator(p), directory_iterator(), back_inserter(dir_elements));
    for(vec::const_iterator it=dir_elements.begin(); it != dir_elements.end(); ++it)
        if (is_directory(*it))
        {
            vec dir_elements_within;
            copy(directory_iterator(*it), directory_iterator(), back_inserter(dir_elements_within));
            elements_number += dir_elements_within.size();
        }
    LOG(INFO) << "Counting is finished. " << elements_number << " files are found." << std::endl;
    return elements_number;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "Usage: bmp_converter <path> <target_set> <db>\n";
        return 1;
    }

    path p (argv[1]);
    int target = argToClass(argv[2][0]);
    CHECK_NE(target, -1)
            << "Target set is incorrect. Use d, c, s instead\n";
    LOG(INFO) << "Target set is: " << classToString(target) << "\n";
    TARGET_SET = static_cast<LABEL_SET> (target);

    if (exists(p))    // does p actually exist?
    {

      if (is_regular_file(p))        // is p a regular file?
        cout << "<path> should be directory not a file" << '\n';

      else if (is_directory(p))      // is p a directory?
      {

        shared_ptr<LMDB_DESCRIPTOR> lmdb(new LMDB_DESCRIPTOR()); // single lmdb for whole program
        vec dir_elements;

        char *pixels, *db_path = argv[3];
        string value;

        LOG(INFO) << "Opening lmdb " << db_path;
        CHECK_EQ(mkdir(db_path, 0744), 0)
            << "mkdir " << db_path << "failed";
        CHECK_EQ(mdb_env_create(&lmdb->mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(lmdb->mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
        CHECK_EQ(mdb_env_open(lmdb->mdb_env, db_path, 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
        CHECK_EQ(mdb_txn_begin(lmdb->mdb_env, NULL, 0, &lmdb->mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_open(lmdb->mdb_txn, NULL, 0, &lmdb->mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";

        lmdb->files_number = getFilesNumber(p);
        copy(directory_iterator(p), directory_iterator(), back_inserter(dir_elements));
        vec::const_iterator it (dir_elements.begin());
        while(it != dir_elements.end())
        {
            boost::thread_group thread_group;
            for(int thread_num = 0; thread_num < MAXIMUM_NUMBER_OF_THREADS && it != dir_elements.end(); ++thread_num)
            {
                boost::thread* current_thread = new boost::thread(lmdbThread, lmdb.get(), &(*it));  //thread group frees all thread objects after destruction
                thread_group.add_thread(current_thread); // so there is no need to manage this process
                it++; // http://www.boost.org/doc/libs/1_35_0/doc/html/thread/thread_management.html#thread.thread_management.threadgroup.destructor
            }
            thread_group.join_all();
        }

        //close db
        CHECK_EQ(mdb_txn_commit(lmdb->mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(lmdb->mdb_env, NULL, 0, &lmdb->mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
        CHECK_EQ(mdb_txn_commit(lmdb->mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(lmdb->mdb_env, lmdb->mdb_dbi);
        mdb_env_close(lmdb->mdb_env);

        LOG(INFO) << lmdb->increaseItemsCounter() << " items have been processed and stored to the database." << std::endl;
      }
      else
        cout << p << " exists, but is neither a regular file nor a directory\n" << std::endl;
    }


  return 0;
}
