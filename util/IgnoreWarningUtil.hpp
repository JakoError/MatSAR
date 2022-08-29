//
// Created by JakoError on 2022/7/6.
//

#ifndef MATSAR_IGNOREWARNINGUTIL_HPP
#define MATSAR_IGNOREWARNINGUTIL_HPP

/**
 * warning C4244: “初始化”: 从“const dType”转换到“_Ty”，可能丢失数据
 * 用户进行强转时会出现的报错
 */
#pragma warning (disable: 4244 4190)

#endif //MATSAR_IGNOREWARNINGUTIL_HPP
