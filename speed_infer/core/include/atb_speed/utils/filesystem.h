/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_UTILS_FILESYSTEM_H
#define ATB_SPEED_UTILS_FILESYSTEM_H
#include <string>
#include <vector>

namespace atb_speed {
class FileSystem {
public:
    static void GetDirChildItems(const std::string &dirPath, std::vector<std::string> &itemPaths);
    static void GetDirChildFiles(const std::string &dirPath, std::vector<std::string> &filePaths);
    static void GetDirChildDirs(const std::string &dirPath, std::vector<std::string> &dirPaths);
    static bool Exists(const std::string &path);
    static bool IsDir(const std::string &path);
    static std::string Join(const std::vector<std::string> &paths);
    static int64_t FileSize(const std::string &filePath);
    static std::string BaseName(const std::string &filePath);
    static std::string DirName(const std::string &path);
    static bool ReadFile(const std::string &filePath, char *buffer, uint64_t bufferSize);
    static bool WriteFile(const void *codeBuf, uint64_t codeLen, const std::string &filePath);
    static void DeleteFile(const std::string &filePath);
    static bool Rename(const std::string &filePath, const std::string &newFilePath);
    static bool MakeDir(const std::string &dirPath, int mode);
    static bool Makedirs(const std::string &dirPath, const mode_t mode);

private:
    static void GetDirChildItemsImpl(const std::string &dirPath, bool matchFile, bool matchDir,
                                     std::vector<std::string> &itemPaths);
};
} // namespace atb_speed
#endif