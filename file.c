// Copyright © 2005-2007 Jens Gulden
// Copyright © 2011-2011 Diego Elio Pettenò
// SPDX-FileCopyrightText: 2005 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

/* --- tool functions for file handling ------------------------------------ */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>

#include "imageprocess/blit.h"
#include "unpaper.h"

/**
 * Loads image data from a file in pnm format.
 *
 * @param f file to load
 * @param image structure to hold loaded image
 * @param type returns the type of the loaded image
 */
void loadImage(const char *filename, Image *image, Pixel sheet_background,
               uint8_t abs_black_threshold) {
  int ret;
  AVFormatContext *s = NULL;
  AVCodecContext *avctx = NULL;
  const AVCodec *codec;
  AVPacket pkt;
  AVFrame *frame = av_frame_alloc();
  char errbuff[1024];

  ret = avformat_open_input(&s, filename, NULL, NULL);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("unable to open file %s: %s", filename, errbuff);
  }

  avformat_find_stream_info(s, NULL);

  if (verbose >= VERBOSE_MORE)
    av_dump_format(s, 0, filename, 0);

  if (s->nb_streams < 1)
    errOutput("unable to open file %s: missing streams", filename);

  codec = avcodec_find_decoder(s->streams[0]->codecpar->codec_id);
  if (!codec)
    errOutput("unable to open file %s: unsupported format", filename);

  avctx = avcodec_alloc_context3(codec);
  if (!avctx)
    errOutput("cannot allocate decoder context for %s", filename);

  ret = avcodec_parameters_to_context(avctx, s->streams[0]->codecpar);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof errbuff);
    errOutput("unable to copy parameters to context: %s", errbuff);
  }

  ret = avcodec_open2(avctx, codec, NULL);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof errbuff);
    errOutput("unable to open file %s: %s", filename, errbuff);
  }

  ret = av_read_frame(s, &pkt);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof errbuff);
    errOutput("unable to open file %s: %s", filename, errbuff);
  }

  if (pkt.stream_index != 0)
    errOutput("unable to open file %s: invalid stream.", filename);

  ret = avcodec_send_packet(avctx, &pkt);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof errbuff);
    errOutput("cannot send packet to decoder: %s", errbuff);
  }

  ret = avcodec_receive_frame(avctx, frame);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof errbuff);
    errOutput("error while receiving frame from decoder: %s", errbuff);
  }

  Rectangle area = rectangle_from_size(
      POINT_ORIGIN,
      (RectangleSize){.width = frame->width, .height = frame->height});

  switch (frame->format) {
  case AV_PIX_FMT_Y400A: // 8-bit grayscale PNG
  case AV_PIX_FMT_GRAY8:
  case AV_PIX_FMT_RGB24:
  case AV_PIX_FMT_MONOBLACK:
  case AV_PIX_FMT_MONOWHITE:
    *image = create_image(size_of_rectangle(area), frame->format, false,
                          sheet_background, abs_black_threshold);
    av_frame_free(&image->frame);
    image->frame = av_frame_clone(frame);
    break;

  case AV_PIX_FMT_PAL8: {
    *image = create_image(size_of_rectangle(area), AV_PIX_FMT_RGB24, false,
                          sheet_background, abs_black_threshold);

    const uint32_t *palette = (const uint32_t *)frame->data[1];
    scan_rectangle(area) {
      const uint8_t palette_index = frame->data[0][frame->linesize[0] * y + x];
      set_pixel(*image, (Point){x, y},
                pixel_from_value(palette[palette_index]));
    }
  } break;

  default:
    errOutput("unable to open file %s: unsupported pixel format", filename);
  }

  avcodec_free_context(&avctx);
  avformat_close_input(&s);
}

/**
 * Fast direct PNM writer - bypasses FFmpeg for simple formats.
 * Returns true if handled, false to fall back to FFmpeg.
 */
static bool saveImageDirect(const char *filename, Image output) {
  FILE *f = fopen(filename, "wb");
  if (!f)
    return false;

  int width = output.frame->width;
  int height = output.frame->height;
  int format = output.frame->format;

  switch (format) {
  case AV_PIX_FMT_GRAY8: {
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
      fwrite(output.frame->data[0] + y * output.frame->linesize[0], 1, width,
             f);
    }
    break;
  }
  case AV_PIX_FMT_RGB24: {
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
      fwrite(output.frame->data[0] + y * output.frame->linesize[0], 1,
             width * 3, f);
    }
    break;
  }
  case AV_PIX_FMT_MONOWHITE: {
    fprintf(f, "P4\n%d %d\n", width, height);
    int row_bytes = (width + 7) / 8;
    for (int y = 0; y < height; y++) {
      const uint8_t *src =
          output.frame->data[0] + y * output.frame->linesize[0];
      fwrite(src, 1, row_bytes, f);
    }
    break;
  }
  default:
    fclose(f);
    return false;
  }

  fclose(f);
  return true;
}

/**
 * Saves image data to a file in pgm or pbm format.
 *
 * @param filename file name to save image to
 * @param image image to save
 * @param type filetype of the image to save
 * @return true on success, false on failure
 */
void saveImage(char *filename, Image input, int outputPixFmt) {
  Image output = input;

#if defined(UNPAPER_WITH_CUDA) && (UNPAPER_WITH_CUDA)
  image_ensure_cpu(&input);
#endif

  switch (outputPixFmt) {
  case AV_PIX_FMT_Y400A:
    outputPixFmt = AV_PIX_FMT_GRAY8;
    break;
  case AV_PIX_FMT_MONOBLACK:
    outputPixFmt = AV_PIX_FMT_MONOWHITE;
    break;
  }

  if (input.frame->format != outputPixFmt) {
    output = create_image(size_of_image(input), outputPixFmt, false,
                          input.background, input.abs_black_threshold);

    int width = input.frame->width;
    int height = input.frame->height;
    bool fast_path = false;

    if (input.frame->format == AV_PIX_FMT_RGB24 &&
        outputPixFmt == AV_PIX_FMT_MONOWHITE) {
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input.frame->data[0] + y * input.frame->linesize[0];
        uint8_t *dst = output.frame->data[0] + y * output.frame->linesize[0];
        for (int x = 0; x < width; x++) {
          int gray = (src[x * 3] + src[x * 3 + 1] + src[x * 3 + 2]) / 3;
          int bit_index = x % 8;
          if (bit_index == 0)
            dst[x / 8] = 0;
          if (gray < input.abs_black_threshold)
            dst[x / 8] |= (0x80 >> bit_index);
        }
      }
      fast_path = true;
    } else if (input.frame->format == AV_PIX_FMT_GRAY8 &&
               outputPixFmt == AV_PIX_FMT_MONOWHITE) {
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input.frame->data[0] + y * input.frame->linesize[0];
        uint8_t *dst = output.frame->data[0] + y * output.frame->linesize[0];
        for (int x = 0; x < width; x++) {
          int bit_index = x % 8;
          if (bit_index == 0)
            dst[x / 8] = 0;
          if (src[x] < input.abs_black_threshold)
            dst[x / 8] |= (0x80 >> bit_index);
        }
      }
      fast_path = true;
    } else if (input.frame->format == AV_PIX_FMT_MONOBLACK &&
               outputPixFmt == AV_PIX_FMT_MONOWHITE) {
      int row_bytes = (width + 7) / 8;
      for (int y = 0; y < height; y++) {
        const uint8_t *src =
            input.frame->data[0] + y * input.frame->linesize[0];
        uint8_t *dst = output.frame->data[0] + y * output.frame->linesize[0];
        for (int x = 0; x < row_bytes; x++) {
          dst[x] = src[x] ^ 0xFF;
        }
      }
      fast_path = true;
    }

    if (!fast_path) {
      copy_rectangle_cpu(input, output, full_image(input), POINT_ORIGIN);
    }
  }

  if (saveImageDirect(filename, output)) {
    if (output.frame != input.frame)
      av_frame_free(&output.frame);
    return;
  }

  enum AVCodecID output_codec = -1;
  const AVCodec *codec;
  AVFormatContext *out_ctx;
  AVCodecContext *codec_ctx;
  AVStream *video_st;
  AVPacket *pkt = NULL;
  int ret;
  char errbuff[1024];

  if (avformat_alloc_output_context2(&out_ctx, NULL, "image2", filename) < 0 ||
      out_ctx == NULL) {
    errOutput("unable to allocate output context.");
  }

  if ((ret = av_opt_set(out_ctx->priv_data, "update", "true", 0)) < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("unable to configure update option: %s", errbuff);
  }

  switch (outputPixFmt) {
  case AV_PIX_FMT_RGB24:
    output_codec = AV_CODEC_ID_PPM;
    break;
  case AV_PIX_FMT_GRAY8:
    output_codec = AV_CODEC_ID_PGM;
    break;
  case AV_PIX_FMT_MONOWHITE:
    output_codec = AV_CODEC_ID_PBM;
    break;
  default:
    output_codec = -1;
    break;
  }

  codec = avcodec_find_encoder(output_codec);
  if (!codec) {
    errOutput("output codec not found");
  }

  video_st = avformat_new_stream(out_ctx, codec);
  if (!video_st) {
    errOutput("could not alloc output stream");
  }

  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    errOutput("could not alloc codec context");
  }

  codec_ctx->width = output.frame->width;
  codec_ctx->height = output.frame->height;
  codec_ctx->pix_fmt = output.frame->format;
  video_st->codecpar->width = output.frame->width;
  video_st->codecpar->height = output.frame->height;
  video_st->codecpar->format = output.frame->format;
  video_st->time_base.den = codec_ctx->time_base.den = 1;
  video_st->time_base.num = codec_ctx->time_base.num = 1;

  ret = avcodec_open2(codec_ctx, codec, NULL);

  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("unable to open codec: %s", errbuff);
  }

  if (verbose >= VERBOSE_MORE)
    av_dump_format(out_ctx, 0, filename, 1);

  if ((ret = avio_open(&out_ctx->pb, filename, AVIO_FLAG_WRITE)) < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("cannot alloc I/O context for %s: %s", filename, errbuff);
  }

  if ((ret = avformat_write_header(out_ctx, NULL)) < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("error writing header to '%s': %s", filename, errbuff);
  }

  pkt = av_packet_alloc();
  if (!pkt) {
    errOutput("unable to allocate output packet");
  }

  ret = avcodec_send_frame(codec_ctx, output.frame);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("unable to send frame to encoder: %s", errbuff);
  }

  ret = avcodec_receive_packet(codec_ctx, pkt);
  if (ret < 0) {
    av_strerror(ret, errbuff, sizeof(errbuff));
    errOutput("unable to receive packet from encoder: %s", errbuff);
  }

  av_write_frame(out_ctx, pkt);

  av_write_trailer(out_ctx);

  av_packet_free(&pkt);
  avcodec_free_context(&codec_ctx);
  avformat_free_context(out_ctx);

  if (output.frame != input.frame)
    av_frame_free(&output.frame);
}

/**
 * Saves the image if full debugging mode is enabled.
 */
void saveDebug(char *filenameTemplate, int index, Image image) {
  if (verbose >= VERBOSE_DEBUG_SAVE) {
    char debugFilename[100];
    sprintf(debugFilename, filenameTemplate, index);
    saveImage(debugFilename, image, image.frame->format);
  }
}

/**
 * Detect pixel format from file extension.
 *
 * @param filename file name to examine
 * @return AVPixelFormat corresponding to the extension, or AV_PIX_FMT_NONE
 */
int detectPixelFormatFromExtension(const char *filename) {
  if (filename == NULL) {
    return AV_PIX_FMT_NONE;
  }
  size_t len = strlen(filename);
  if (len < 4) {
    return AV_PIX_FMT_NONE;
  }
  const char *ext = filename + len - 4;
  if (strcasecmp(ext, ".pbm") == 0) {
    return AV_PIX_FMT_MONOWHITE;
  }
  if (strcasecmp(ext, ".pgm") == 0) {
    return AV_PIX_FMT_GRAY8;
  }
  if (strcasecmp(ext, ".ppm") == 0) {
    return AV_PIX_FMT_RGB24;
  }
  return AV_PIX_FMT_NONE;
}
