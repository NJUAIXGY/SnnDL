# -*- Makefile -*-
#
# SnnDL元素库构建配置
#

AM_CPPFLAGS += \
	$(MPI_CPPFLAGS) \
	-I$(top_srcdir)/src

compdir = $(pkglibdir)
comp_LTLIBRARIES = libSnnDL.la

# 源文件列表
libSnnDL_la_SOURCES = \
	SpikeEvent.h \
	SnnPE.h \
	SnnPE.cc \
	SpikeSource.h \
	SpikeSource.cc \
	SnnInterface.h \
	SnnNIC.h \
	SnnNIC.cc \
	SnnInterface.h \
	SnnNIC.h \
	SnnNIC.cc

# 编译器标志
libSnnDL_la_CPPFLAGS = \
	$(BOOST_CPPFLAGS) \
	$(AM_CPPFLAGS)

# 链接器标志
libSnnDL_la_LDFLAGS = \
	-module \
	-avoid-version \
	$(BOOST_LDFLAGS)

# 安装头文件
nobase_include_HEADERS = \
	SpikeEvent.h \
	SnnPE.h \
	SpikeSource.h

# 可选的HDF5支持（如果系统有HDF5库）
if HAVE_HDF5
libSnnDL_la_CPPFLAGS += $(HDF5_CPPFLAGS)
libSnnDL_la_LDFLAGS += $(HDF5_LDFLAGS)
libSnnDL_la_LIBADD = $(HDF5_LIBS)
endif

# 文档和示例文件
EXTRA_DIST = \
	README.md

# 清理目标
CLEANFILES = *.la *.lo *.o .libs/*
