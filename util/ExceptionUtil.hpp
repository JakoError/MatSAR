//
// Created by JakoError on 2022/7/3.
//

#ifndef MATSAR_EXCEPTIONUTIL_HPP
#define MATSAR_EXCEPTIONUTIL_HPP

#ifndef MATSAR_TYPEUTIL_HPP

#include <cassert>
#include <cstdlib>
#include <utility>
#include <string>

#include "TypeUtil.hpp"

#endif

#define RUNTIME_ERROR_CODE (-1)
#define LOGICAL_ERROR_CODE (-2)
//#define PROPERTY_EXCEPTION_CODE -3
#define OUT_OF_RANGE_CODE (-4)
#define VALUE_ERROR_CODE (-5)
#define TYPE_LOSS_EXCEPTION_CODE (-6)
#define SHAPE_EXCEPTION_CODE (-7)
#define BROADCAST_EXCEPTION_CODE (-8)

using namespace std;

namespace MatSAR::Util::Exception {
    /**
     * 数据不匹配等问题
     */
    class ValueError : logic_error {
    public:
        using Mybase = logic_error;

        explicit ValueError(const string &Message) : Mybase(Message.c_str()) {}

        explicit ValueError(const char *Message) : Mybase(Message) {}

        [[nodiscard]] auto what() const noexcept -> const char * override {
            return std::logic_error::what();
        }
    };

    /**
     * 矩阵shape不匹配或出错等问题
     */
    class ShapeException : logic_error {
    public:
        using Mybase = logic_error;

        explicit ShapeException(const string &Message) : Mybase(Message.c_str()) {}

        explicit ShapeException(const char *Message) : Mybase(Message) {}

        [[nodiscard]] auto what() const noexcept-> const char * override {
            return std::logic_error::what();
        }
    };

    /**
     * 因数据类型不一致出现的数据丢失
     */
    class TypeLossException : logic_error {
    public:
        using Mybase = logic_error;

        explicit TypeLossException(const string &Message) : Mybase(Message.c_str()) {}

        explicit TypeLossException(const char *Message) : Mybase(Message) {}

        [[nodiscard]] auto what() const noexcept-> const char * override {
            return std::logic_error::what();
        }
    };

    /**
     * 因数据类型不一致出现的数据丢失
     */
    class BroadcastException : logic_error {
    public:
        using Mybase = logic_error;

        explicit BroadcastException(const string &Message) : Mybase(Message.c_str()) {}

        explicit BroadcastException(const char *Message) : Mybase(Message) {}

        [[nodiscard]] auto what() const noexcept-> const char * override {
            return std::logic_error::what();
        }
    };

    void throw_value_error(const string &msg, bool to_exit = true) {
        try {
            throw ValueError(msg);
        } catch (ValueError &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "ValueError:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(VALUE_ERROR_CODE);
        }
    }

    void throw_logical_error(const string &msg, bool to_exit = true) {
        try {
            throw logic_error(msg);
        } catch (logic_error &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "LogicalError:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(LOGICAL_ERROR_CODE);
        }
    }

    void throw_out_of_range_error(const string &msg, bool to_exit = true) {
        try {
            throw out_of_range(msg);
        } catch (out_of_range &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "out_of_range:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(OUT_OF_RANGE_CODE);
        }
    }

    template<Calculable T1, Calculable T2>
    void throw_type_loss_exception(bool to_exit = false) {
        try {
            string msg = "convert from \"" + MatSAR::Util::TypeUtil::getCalculableString<T1>() + "\" to \"" + MatSAR::Util::TypeUtil::getCalculableString<T2>() +
                         "\" may loss data";
            throw TypeLossException(msg);
        } catch (TypeLossException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "TypeLossException:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(TYPE_LOSS_EXCEPTION_CODE);
        }
    }

    void throw_broadcast_exception(const string &msg, bool to_exit = true) {
        try {
            throw BroadcastException(msg);
        } catch (BroadcastException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "BroadcastException:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(BROADCAST_EXCEPTION_CODE);
        }
    }

    void throw_broadcast_exception(const string &shape_str, const string &target_str, bool to_exit = true) {
        try {
            throw BroadcastException("broadcast failed: cannot broadcast " + shape_str +
                                     " to " + target_str);
        } catch (BroadcastException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "BroadcastException:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(BROADCAST_EXCEPTION_CODE);
        }
    }

    void throw_shape_exception(const string &msg, bool to_exit = true) {
        try {
            throw ShapeException(msg);
        } catch (ShapeException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "ShapeException:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(SHAPE_EXCEPTION_CODE);
        }
    }

    void throw_shape_exception(const string &pre_msg, const string &shape1_str, const string &shape2_str,
                               bool to_exit = true) {
        try {
            stringstream ss;
            ss << pre_msg << "Matrix shape not match: shape1 = " << shape1_str << " shape2 = " << shape2_str;
            throw ShapeException(ss.str());
        } catch (ShapeException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "ShapeException:" << e.what() << endl;
            flush(cerr);
            if (to_exit)
                exit(SHAPE_EXCEPTION_CODE);
        }
    }

    /**
     * 无条件退出!
     */
    void throw_runtime_error_exception(const string &msg) {
        try {
            throw runtime_error(msg);
        } catch (BroadcastException &e) {
            fflush(stdout);
            flush(cerr);
            cerr << "RuntimeError(debug):" << e.what() << endl;
            flush(cerr);
            exit(RUNTIME_ERROR_CODE);
        }
    }

    void throw_warning(const string &msg) {
        fflush(stdout);
        flush(cerr);
        cerr << "WARNING:" << msg << endl;
        flush(cerr);
    }

}
#endif //MATSAR_EXCEPTIONUTIL_HPP