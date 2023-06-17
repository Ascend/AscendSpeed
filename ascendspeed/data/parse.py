# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import argparse


class ParseDataPaths(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):

        if option_string == "--train-weighted-split-paths":
            assert len(values) == 1, 'Only 1 dataset group is allowed to'
            'be passed for the argument --train-weighted-split-paths'

        # make sure string given in the correct format
        err_message = 'Each data group should be input on the following format'
        '"GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2"'
        'where START < END'
        for v in values:
            # each prefix consists several datasets separated by commas
            prefix = ":".join(v.split(":")[1:]) # remove GIVEN_NAME
            datasets = prefix.split(",")
            # check if each dataset is formatted like `WEIGHT START:END PATH`
            for d in datasets:
                assert len(d.split()) == 3, err_message
                start, end = d.split()[1].split(":")
                assert float(start) < float(end), err_message

        names = [v.split(":")[0] for v in values]

        prefixes = [":".join(v.split(":")[1:]).strip() for v in values]
        weights = [[d.split()[0] for d in p.split(",")] for p in prefixes]
        splits = [[d.split()[1] for d in p.split(",")] for p in prefixes]
        paths = [[d.split()[2] for d in p.split(",")] for p in prefixes]

        # # to keep consistency with Option 1 of data loading (through --data-path)
        # #  paths will contain strings on the following form
        # # "WEIGHTS1 PATH1 WEIGHTS2 PATH2 WEIGHTS3 PATH3" for each dataset group
        # # while data will be parsed in additional arguments below
        setattr(args, self.dest, paths)
        setattr(args, self.dest.replace("paths", "weights"), weights)
        setattr(args, self.dest.replace("paths", "splits"), splits)
        setattr(args, self.dest.replace("paths","names"), names)


class ParseDataPathsPath(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        expected_option_strings = ["--train-weighted-split-paths-path", 
            "--valid-weighted-split-paths-path", "--test-weighted-split-paths-path"]
        assert option_string in expected_option_strings, \
            f"Expected {option_string} to be in {expected_option_strings}"

        with open(values, "r") as fi:
            lines = fi.readlines()
            assert len(lines) == 1, f"Got multiple lines {len(lines)} instead of 1 expected"
            assert lines[0][-2:] == "\"\n" and \
                lines[0][0] == "\"", f"Invalid input format, got {lines}"
            values = lines[0][1:-2].split("\" \"")
            weighted_split_paths_dest = re.sub(r"_path$", "", self.dest)
            weighted_split_paths_option = re.sub(r"-path$", "", self.option_strings[0])
            setattr(args, weighted_split_paths_dest, values)
            ParseDataPaths(option_strings=[weighted_split_paths_option], 
                dest=weighted_split_paths_dest)(parser, args, values, option_string=weighted_split_paths_option)