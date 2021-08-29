/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIPSAMPLE_H_
#define HIPSAMPLE_H_

/******************************************************************************
* Included header files                                                       *
******************************************************************************/

#include "SDKUtil.hpp"

namespace appsdk
{

class HIPCommandArgs: public SDKCmdArgsParser
{
    public:
        unsigned int deviceId;       /**< Cmd Line Option- if deviceId */
        bool enableDeviceId;         /**< Cmd Line Option- if enableDeviceId */

        /**
        ***********************************************************************
        * @fn initialize
        * @brief Initialize the resources used by tests
        * @return 0 on success Positive if expected and Non-zero on failure
        **********************************************************************/
        int initialize()
        {
            int defaultOptions = 5;
            Option *optionList = new Option[defaultOptions];
            CHECK_ALLOCATION(optionList, "Error. Failed to allocate memory (optionList)\n");
            optionList[0]._sVersion = "q";
            optionList[0]._lVersion = "quiet";
            optionList[0]._description = "Quiet mode. Suppress all text output.";
            optionList[0]._type = CA_NO_ARGUMENT;
            optionList[0]._value = &quiet;
            optionList[1]._sVersion = "e";
            optionList[1]._lVersion = "verify";
            optionList[1]._description = "Verify results against reference implementation.";
            optionList[1]._type = CA_NO_ARGUMENT;
            optionList[1]._value = &verify;
            optionList[2]._sVersion = "t";
            optionList[2]._lVersion = "timing";
            optionList[2]._description = "Print timing.";
            optionList[2]._type = CA_NO_ARGUMENT;
            optionList[2]._value = &timing;
            optionList[3]._sVersion = "v";
            optionList[3]._lVersion = "version";
            optionList[3]._description = "AMD APP SDK version string.";
            optionList[3]._type = CA_NO_ARGUMENT;
            optionList[3]._value = &version;
            optionList[4]._sVersion = "d";
            optionList[4]._lVersion = "deviceId";
            optionList[4]._description =
                "Select deviceId to be used[0 to N-1 where N is number devices available].";
            optionList[4]._type = CA_ARG_INT;
            optionList[4]._value = &deviceId;
            _numArgs = defaultOptions;
            _options = optionList;
            return SDK_SUCCESS;
        }

        /**
        ***********************************************************************
        * @brief Destroy the resources used by tests
        **********************************************************************/
        virtual ~HIPCommandArgs()
        {
        }

        /**
        ***********************************************************************
        * @brief Constructor, initialize the resources used by tests
        * @param sampleName Name of the Sample
        **********************************************************************/
        HIPCommandArgs()
        {
            deviceId = 0;
            enableDeviceId = false;
        }

        /**
        ***********************************************************************
        * @brief parseCommandLine parses the command line options given by user
        * @param argc Number of elements in cmd line input
        * @param argv array of char* storing the CmdLine Options
        * @return 0 on success Positive if expected and Non-zero on failure
        **********************************************************************/
        int parseCommandLine(int argc, char **argv)
        {
            if(!parse(argv,argc))
            {
                usage();
                if(isArgSet("h",true) == true)
                {
                    exit(SDK_SUCCESS);
                }
                return SDK_FAILURE;
            }
            if(isArgSet("h",true) == true)
            {
                usage();
                exit(SDK_SUCCESS);
            }
            if(isArgSet("v", true)
                    || isArgSet("version", false))
            {
                std::cout << "SDK version : " << sampleVerStr.c_str()
                          << std::endl;
                exit(0);
            }
            if(isArgSet("d",true)
                    || isArgSet("deviceId",false))
            {
                enableDeviceId = true;
            }
            return SDK_SUCCESS;
        }
};

}
#endif
