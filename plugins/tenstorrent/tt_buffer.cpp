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

nxs_long cdiv(nxs_long size, nxs_long incr) {
  return (size + incr -1) / incr;
}

TTBuffer::TTBuffer(TTDevice *dev, nxs_buffer_layout shape,
                   void *data_ptr, nxs_uint settings)
  : Buffer(shape, data_ptr, settings), device(dev) {
    if (shape.rank != 0) {
      // Pad up to the nearest tile size
      rowCount = 1;
      paddedSize = 1;
      tilizedShape.rank = shape.rank;
      if (shape.rank == 1) {
        // make 2D
        tilizedShape.rank = 2;
        tilizedShape.dim[0] = cdiv(shape.dim[0], tileWidth * tileWidth) * tileWidth;
        rowCount = tilizedShape.dim[1] = tileWidth;
        paddedSize = tilizedShape.dim[0] * tilizedShape.dim[1];
      } else {
        for (nxs_uint i = 0; i < shape.rank; i++) {
          tilizedShape.dim[i] = cdiv(shape.dim[i], tileWidth) * tileWidth;
          paddedSize *= tilizedShape.dim[i];
          if (i != 0) rowCount *= tilizedShape.dim[i];
        }
      }

      // Size of a tile in bytes
      elementSize = getElementSizeBits() / 8;
      nxs_ulong tileSizeBytes = tileWidth * tileWidth * elementSize;

      // Create buffer in DRAM.
      NXSLOG_INFO("TTBuffer: tile_size={} size={}", tileSizeBytes, paddedSize);
      ttmd::DeviceLocalBufferConfig dram_config{
          .page_size = tileSizeBytes,  // Number of bytes when round-robin between banks. Usually this is the same
                                        // as the tile size for efficiency.
          .buffer_type = ttm::BufferType::DRAM};  // Type of buffer (DRAM or L1(SRAM))
      ttmd::ReplicatedBufferConfig distributed_buffer_config{
          .size = paddedSize * elementSize  // Size of the buffer in bytes
      };
      TT_OBJ_CHECK(buffer, ttmd::MeshBuffer::create, distributed_buffer_config, dram_config, device->get().get());
      address = buffer->address();

      if (!(getSettings() & NXS_BufferSettings_OnDevice) && data_ptr != nullptr) {
        setSettings(getSettings() | NXS_BufferSettings_OnDevice);
        copyToDevice(getData(), false);
      }
    }
}

template <typename T>
nxs_status TTBuffer::tilizeAndCopyToDevice(T *data_ptr, bool blocking) {
  nxs_ulong tilizedStride = tilizedShape.dim[0];

  auto shape = getShape();
  nxs_ulong rowStride = shape.dim[0];
  std::vector<T> buf_v(paddedSize, 0);
  if (shape.rank == 1) {
    // copy entire input flat
    std::copy(data_ptr, data_ptr + rowStride, buf_v.begin());
  } else {
    for (nxs_ulong i = 0; i < rowCount; i++) {
      std::copy(data_ptr, data_ptr + rowStride, buf_v.begin() + i * tilizedStride);
      data_ptr += rowStride;
    }
  }
  buf_v = tilize_nfaces(buf_v, tilizedShape.dim[0], rowCount);

  auto &cq = device->getCQ();
  TT_CHECK(ttmd::EnqueueWriteMeshBuffer, cq, buffer, buf_v, blocking);
  // either buf_v must be held stable or this must finish
  //if (blocking) {
  TT_CHECK(ttmd::Finish, cq);
  //}

  NXSLOG_INFO("TTBuffer: tilizeAndCopyToDevice: address={} size={}", address,
             paddedSize);
  return NXS_Success;
}

nxs_status TTBuffer::copyToDevice(void *host_buf, bool blocking) {
  nxs_status ret = NXS_Success;
  switch (getDataType()) {
    case NXS_DataType_F32:
      ret = tilizeAndCopyToDevice<float>(reinterpret_cast<float *>(host_buf), blocking);
      break;
    case NXS_DataType_BF16:
      ret = tilizeAndCopyToDevice<bfloat16>(reinterpret_cast<bfloat16 *>(host_buf), blocking);
      break;
    case NXS_DataType_U32:
      ret = tilizeAndCopyToDevice<uint32_t>(reinterpret_cast<uint32_t *>(host_buf), blocking);
      break;
    case NXS_DataType_U16:
      ret = tilizeAndCopyToDevice<uint16_t>(reinterpret_cast<uint16_t *>(host_buf), blocking);
      break;
    default:
      NXSLOG_ERROR("TTBuffer: Unsupported data type: {}", nxsGetDataTypeName(getDataType()));
      ret = NXS_InvalidDataType;
      break;
  }
  if (ret == NXS_Success)
    address = buffer->address(); // defer until cq finished
  return NXS_Success;
}

template <typename T>
nxs_status TTBuffer::copyToHostUntilize(T *data_ptr) {
  NXSLOG_INFO("TTBuffer: copyToHostUntilize: address={} size={}", address,
             paddedSize);
  std::vector<T> buf_v(paddedSize);
  auto &cq = device->getCQ();
  TT_CHECK(ttmd::EnqueueReadMeshBuffer, cq, buf_v, buffer, true);
  TT_CHECK(ttmd::Finish, cq);

  nxs_ulong tilizedStride = tilizedShape.dim[0];
  buf_v = untilize_nfaces(buf_v, tilizedStride, rowCount);

  auto shape = getShape();
  T *tbuf_ptr = reinterpret_cast<T *>(buf_v.data());
  nxs_ulong rowStride = shape.dim[0];
  if (shape.rank == 1) {
    // copy entire input flat
    std::copy(data_ptr, data_ptr + rowStride, buf_v.begin());
  } else {
    nxs_ulong tilizedRowStride = tilizedStride * elementSize;
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
        return copyToHostUntilize<float>(reinterpret_cast<float *>(host_buf));
        break;
      case NXS_DataType_BF16:
        return copyToHostUntilize<bfloat16>(reinterpret_cast<bfloat16 *>(host_buf));
        break;
      case NXS_DataType_U32:
        return copyToHostUntilize<uint32_t>(reinterpret_cast<uint32_t *>(host_buf));
        break;
      case NXS_DataType_U16:
        return copyToHostUntilize<uint16_t>(reinterpret_cast<uint16_t *>(host_buf));
        break;
      default:
        NXSLOG_ERROR("TTBuffer: Unsupported data type: {}", nxsGetDataTypeName(getDataType()));
        break;
    }
  }
  return NXS_Success;
}

