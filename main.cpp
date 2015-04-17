#include <iostream>
#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;
using namespace boost::filesystem;
using namespace boost::algorithm;

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    cout << "Usage: bmp_converter <path>\n";
    return 1;
  }

  path p (argv[1]);   // p reads clearer than argv[1] in the following code


  try
  {
    if (exists(p))    // does p actually exist?
    {
      if (is_regular_file(p))        // is p a regular file?
        cout << p << " size is " << file_size(p) << '\n';

      else if (is_directory(p))      // is p a directory?
      {
        typedef vector<path> vec;             // store paths,
        vec v;                                // so we can sort them later

        copy(directory_iterator(p), directory_iterator(), back_inserter(v));
        typedef vector< string > split_vector_type;
        split_vector_type SplitVec;

        for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
        {
          split( SplitVec, (*it).string(), is_any_of("\\/"), token_compress_on );
          int i = 0;

          while(i!=SplitVec.size())
          {
              if (SplitVec[i])
          }
          if (is_regular_file(*it))
            cout << SplitVec[SplitVec.size()-1] << '\n';
        }

      }
      else
        cout << p << " exists, but is neither a regular file nor a directory\n";
    }
    else
      cout << p << " does not exist\n";
  }

  catch (const filesystem_error& ex)
  {
    cout << ex.what() << '\n';
  }

  return 0;
}
