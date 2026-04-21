#include <tt_buffer.h>
#include <tt_device.h>
#include <tt_runtime.h>

#include <tt-metalium/tilize_utils.hpp>

static bool compareShapes(nxs_buffer_layout shape1, nxs_buffer_layout shape2) {
  if (shape1.rank != shape2.rank) {
    return false;
  }
  for (nxs_uint i = 0; i < shape1.rank; i++) {
    if (shape1.dim[i] != shape2.dim[i]) {
      return false;
    }
  }
  return true;
}

static nxs_ulong cdiv(nxs_ulong a, nxs_ulong b) {
  return (a + b - 1) / b;
}

TTBuffer::TTBuffer(TTDevice *dev, nxs_buffer_layout shape,
                   void *data_ptr, nxs_uint settings)
  : Buffer(shape, data_ptr, settings), device(dev), address(0) {

  if (shape.rank == 0) {
    NXSLOG_WARN("TTBuffer: empty shape {}", shape.dim[0]);
    return;
  }
  // Pad up to the nearest tile size
  rowCount = 1;
  paddedSize = 1;
  if (shape.rank == 1) {
    // pad up to the nearest tile size
    tilizedShape.rank = 2;
    tilizedShape.dim[0] = cdiv(shape.dim[0], tileWidth * tileWidth) * tileWidth;
    tilizedShape.dim[1] = tileWidth;
    rowCount = tileWidth;
    paddedSize = tilizedShape.dim[0] * tilizedShape.dim[1];
  } else {
    tilizedShape.rank = shape.rank;
    for (nxs_uint i = 0; i < shape.rank; i++) {
      tilizedShape.dim[i] = cdiv(shape.dim[i], tileWidth) * tileWidth;
      NXSLOG_INFO("TTBuffer: dim[{}]={}", i, tilizedShape.dim[i]);
      paddedSize *= tilizedShape.dim[i];
      if (i != 0) rowCount *= tilizedShape.dim[i];
    }
  }
  NXSLOG_INFO("TTBuffer: rank={} padded_size={}", shape.rank, paddedSize);

  // Size of a tile in bytes
  elementSize = getElementSizeBits() / 8;
  nxs_ulong tileSizeBytes = tileWidth * tileWidth * elementSize;

  // Create buffer in DRAM.
  ttmd::DeviceLocalBufferConfig dram_config{
      .page_size = tileSizeBytes,  // Number of bytes when round-robin between banks. Usually this is the same
                                    // as the tile size for efficiency.
      .buffer_type = ttm::BufferType::DRAM};  // Type of buffer (DRAM or L1(SRAM))
  ttmd::ReplicatedBufferConfig distributed_buffer_config{
      .size = paddedSize * elementSize  // Size of the buffer in bytes
  };
  TT_OBJ_CHECK(buffer, ttmd::MeshBuffer::create, distributed_buffer_config, dram_config, device->get().get());
  address = buffer->address();
  NXSLOG_INFO("TTBuffer: tile_size={} padded_size={} address={}", tileSizeBytes, paddedSize, address);

  if (!(getSettings() & NXS_BufferSettings_OnDevice) && data_ptr != nullptr) {
    setSettings(getSettings() | NXS_BufferSettings_OnDevice);
    copyToDevice(getData(), false);
  }
}

template <typename T>
nxs_status TTBuffer::tilizeAndCopyToDevice(T *data_ptr, bool blocking) {
  nxs_ulong tilizedStride = tilizedShape.dim[0];

  auto shape = getShape();
  nxs_ulong rowStride = shape.dim[0];
  std::vector<T> buf_v(paddedSize);
  if (shape.rank == 1) {
    std::copy(data_ptr, data_ptr + getNumElements(), buf_v.begin());
  } else {
    auto *tbuf_ptr = reinterpret_cast<T *>(buf_v.data());
    for (nxs_ulong i = 0; i < rowCount; i++) {
      std::copy(data_ptr, data_ptr + rowStride, tbuf_ptr);
      tbuf_ptr += tilizedStride;
      data_ptr += rowStride;
    }
    buf_v = tilize_nfaces(buf_v, tilizedShape.dim[0], rowCount);
  }

  auto &cq = device->getCQ();
  TT_CHECK(ttmd::EnqueueWriteMeshBuffer, cq, buffer, buf_v, blocking);
  if (blocking) {
    TT_CHECK(ttmd::Finish, cq);
  }

  NXSLOG_INFO("TTBuffer: tilizeAndCopyToDevice: address={} padded_size={}", address,
             paddedSize);
  return NXS_Success;
}

nxs_status TTBuffer::copyToDevice(void *host_buf, bool blocking) {
  switch (getDataType()) {
    case NXS_DataType_F32:
    case NXS_DataType_I32:
    case NXS_DataType_U32:
      return tilizeAndCopyToDevice(reinterpret_cast<uint32_t *>(host_buf), blocking);
      break;
    case NXS_DataType_BF16:
    case NXS_DataType_F16:
    case NXS_DataType_U16:
    case NXS_DataType_I16:
      return tilizeAndCopyToDevice(reinterpret_cast<uint16_t *>(host_buf), blocking);
#if 0
// TODO: tilize_nfaces does not support 8-bit data types
    case NXS_DataType_BF8:
    case NXS_DataType_F8:
    case NXS_DataType_U8:
    case NXS_DataType_I8:
      return tilizeAndCopyToDevice(reinterpret_cast<uint8_t *>(host_buf), blocking);
#endif
    default:
      NXSLOG_ERROR("TTBuffer: Unsupported data type: {}", nxsGetDataTypeName(getDataType()));
      break;
  }
  return NXS_Success;
}

template <typename T>
nxs_status TTBuffer::copyToHostUntilize(T *data_ptr) {
  NXSLOG_INFO("TTBuffer: copyToHostUntilize: address={} size={}", address,
             paddedSize);
  std::vector<T> buf_v(paddedSize);
  auto &cq = device->getCQ();
  TT_CHECK(ttmd::EnqueueReadMeshBuffer, cq, buf_v, buffer, true);

  if (getShape().rank == 1) {
    auto *tbuf_p = buf_v.data();
    std::copy(tbuf_p, tbuf_p + getNumElements(), data_ptr);
  } else {
    nxs_ulong tilizedStride = tilizedShape.dim[0];
    buf_v = untilize_nfaces(buf_v, tilizedStride, rowCount);

    // TODO: handle striding for more than 2 dimensions
    T *tbuf_ptr = reinterpret_cast<T *>(buf_v.data());
    nxs_ulong rowStride = getShape().dim[0];
    nxs_ulong tilizedRowStride = tilizedStride;
    for (nxs_ulong i = 0; i < rowCount; i++) {
      std::copy(tbuf_ptr, tbuf_ptr + rowStride, data_ptr);
      data_ptr += rowStride;
      tbuf_ptr += tilizedRowStride;
    }
  }

  return NXS_Success;
}

nxs_status TTBuffer::copyToHost(void *host_buf) {
  if (buffer) {
    switch (getDataType()) {
      case NXS_DataType_F32:
      case NXS_DataType_U32:
      case NXS_DataType_I32:
        return copyToHostUntilize(reinterpret_cast<uint32_t *>(host_buf));
        break;
      case NXS_DataType_BF16:
      case NXS_DataType_F16:
      case NXS_DataType_U16:
      case NXS_DataType_I16:
        return copyToHostUntilize(reinterpret_cast<uint16_t *>(host_buf));
        break;
#if 0
// TODO: untilize_nfaces does not support 8-bit data types
      case NXS_DataType_BF8:
      case NXS_DataType_F8:
      case NXS_DataType_U8:
      case NXS_DataType_I8:
        return copyToHostUntilize(reinterpret_cast<uint8_t *>(host_buf));
        break;
#endif
    default:
        NXSLOG_ERROR("TTBuffer: Unsupported data type: {}", nxsGetDataTypeName(getDataType()));
        break;
    }
  }
  return NXS_Success;
}

