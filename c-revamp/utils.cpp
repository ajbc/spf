#include "utils.h"

// check if file exisits
bool file_exists(string filename) {
  if ( 0 == access(filename.c_str(), R_OK))
    return true;
  return false;
}

// check if a directory exists
int dir_exists(string dname) {
  struct stat st;
  int ret;

  if (stat(dname.c_str(),&st) != 0) {
    return 0;
  }

  ret = S_ISDIR(st.st_mode);

  /*if(!ret) {
    errno = ENOTDIR;
  }*/

  return ret;
}

void make_directory(string name) {
  mkdir(name.c_str(), S_IRUSR|S_IWUSR|S_IXUSR);
}
