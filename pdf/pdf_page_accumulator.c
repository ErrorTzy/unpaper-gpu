// SPDX-FileCopyrightText: 2025 The unpaper authors
//
// SPDX-License-Identifier: GPL-2.0-only

#include "pdf_page_accumulator.h"
#include "lib/logging.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// Page slot state
typedef enum {
  PAGE_SLOT_EMPTY = 0, // Not yet received
  PAGE_SLOT_PENDING,   // Data received, waiting to be written
  PAGE_SLOT_WRITTEN,   // Successfully written to PDF
  PAGE_SLOT_FAILED,    // Page failed, won't be written
} PageSlotState;

// Page slot in the accumulator
typedef struct {
  PageSlotState state;
  PdfEncodedPage page; // Copy of page data (data pointer is owned)
} PageSlot;

struct PdfPageAccumulator {
  PdfWriter *writer; // PDF writer (not owned)
  int total_pages;   // Total expected pages
  PageSlot *slots;   // Array of page slots

  // Writer state
  int next_write_index; // Next page index to write (sequential)
  atomic_int pages_written;
  atomic_int pages_failed;

  // Synchronization
  pthread_mutex_t mutex;
  pthread_cond_t page_ready; // Signal when a page is ready

  // Writer thread
  pthread_t writer_thread;
  bool writer_running;
  atomic_bool shutdown;
};

// Write a single page to the PDF
static bool write_page_to_pdf(PdfWriter *writer, const PdfEncodedPage *page) {
  switch (page->type) {
  case PDF_PAGE_DATA_JPEG:
    return pdf_writer_add_page_jpeg(writer, page->data, page->data_size,
                                    page->width, page->height, page->dpi);

  case PDF_PAGE_DATA_JP2:
    return pdf_writer_add_page_jp2(writer, page->data, page->data_size,
                                   page->width, page->height, page->dpi);

  case PDF_PAGE_DATA_PIXELS:
    return pdf_writer_add_page_pixels(writer, page->data, page->width,
                                      page->height, page->stride,
                                      page->pixel_format, page->dpi);

  default:
    return false;
  }
}

// Writer thread function - writes pages in sequential order
static void *writer_thread_fn(void *arg) {
  PdfPageAccumulator *acc = (PdfPageAccumulator *)arg;

  while (!atomic_load(&acc->shutdown) ||
         acc->next_write_index < acc->total_pages) {

    pthread_mutex_lock(&acc->mutex);

    // Check if next page is ready
    while (acc->next_write_index < acc->total_pages &&
           acc->slots[acc->next_write_index].state != PAGE_SLOT_PENDING &&
           acc->slots[acc->next_write_index].state != PAGE_SLOT_FAILED &&
           !atomic_load(&acc->shutdown)) {
      // Wait for next page to arrive
      pthread_cond_wait(&acc->page_ready, &acc->mutex);
    }

    // Check if we should exit
    if (atomic_load(&acc->shutdown) &&
        acc->next_write_index >= acc->total_pages) {
      pthread_mutex_unlock(&acc->mutex);
      break;
    }

    // Process consecutive ready pages
    while (acc->next_write_index < acc->total_pages) {
      PageSlot *slot = &acc->slots[acc->next_write_index];

      if (slot->state == PAGE_SLOT_FAILED) {
        // Skip failed page
        verboseLog(VERBOSE_MORE, "Page accumulator: skipping failed page %d\n",
                   acc->next_write_index + 1);
        atomic_fetch_add(&acc->pages_failed, 1);
        acc->next_write_index++;
        continue;
      }

      if (slot->state != PAGE_SLOT_PENDING) {
        // Not ready yet, wait for it
        break;
      }

      // Write the page
      bool success = write_page_to_pdf(acc->writer, &slot->page);

      if (success) {
        verboseLog(VERBOSE_MORE, "Page accumulator: wrote page %d/%d\n",
                   acc->next_write_index + 1, acc->total_pages);
        slot->state = PAGE_SLOT_WRITTEN;
        atomic_fetch_add(&acc->pages_written, 1);
      } else {
        verboseLog(VERBOSE_NORMAL,
                   "Page accumulator: failed to write page %d\n",
                   acc->next_write_index + 1);
        slot->state = PAGE_SLOT_FAILED;
        atomic_fetch_add(&acc->pages_failed, 1);
      }

      // Free the page data
      if (slot->page.data) {
        free(slot->page.data);
        slot->page.data = NULL;
      }

      acc->next_write_index++;
    }

    pthread_mutex_unlock(&acc->mutex);

    // Check if all pages are done
    if (acc->next_write_index >= acc->total_pages) {
      break;
    }
  }

  return NULL;
}

PdfPageAccumulator *pdf_page_accumulator_create(PdfWriter *writer,
                                                int total_pages) {
  if (writer == NULL || total_pages <= 0) {
    return NULL;
  }

  PdfPageAccumulator *acc = calloc(1, sizeof(PdfPageAccumulator));
  if (!acc) {
    return NULL;
  }

  acc->slots = calloc((size_t)total_pages, sizeof(PageSlot));
  if (!acc->slots) {
    free(acc);
    return NULL;
  }

  acc->writer = writer;
  acc->total_pages = total_pages;
  acc->next_write_index = 0;
  atomic_init(&acc->pages_written, 0);
  atomic_init(&acc->pages_failed, 0);
  atomic_init(&acc->shutdown, false);

  pthread_mutex_init(&acc->mutex, NULL);
  pthread_cond_init(&acc->page_ready, NULL);

  // Initialize slots
  for (int i = 0; i < total_pages; i++) {
    acc->slots[i].state = PAGE_SLOT_EMPTY;
    acc->slots[i].page.data = NULL;
  }

  // Start writer thread
  if (pthread_create(&acc->writer_thread, NULL, writer_thread_fn, acc) != 0) {
    pthread_mutex_destroy(&acc->mutex);
    pthread_cond_destroy(&acc->page_ready);
    free(acc->slots);
    free(acc);
    return NULL;
  }

  acc->writer_running = true;

  verboseLog(VERBOSE_MORE, "Page accumulator: created for %d pages\n",
             total_pages);

  return acc;
}

void pdf_page_accumulator_destroy(PdfPageAccumulator *acc) {
  if (!acc) {
    return;
  }

  // Signal shutdown
  atomic_store(&acc->shutdown, true);
  pthread_mutex_lock(&acc->mutex);
  pthread_cond_signal(&acc->page_ready);
  pthread_mutex_unlock(&acc->mutex);

  // Wait for writer thread
  if (acc->writer_running) {
    pthread_join(acc->writer_thread, NULL);
    acc->writer_running = false;
  }

  // Free any remaining page data
  for (int i = 0; i < acc->total_pages; i++) {
    if (acc->slots[i].page.data) {
      free(acc->slots[i].page.data);
    }
  }

  pthread_mutex_destroy(&acc->mutex);
  pthread_cond_destroy(&acc->page_ready);
  free(acc->slots);
  free(acc);
}

bool pdf_page_accumulator_submit(PdfPageAccumulator *acc,
                                 const PdfEncodedPage *page) {
  if (!acc || !page || page->page_index < 0 ||
      page->page_index >= acc->total_pages) {
    return false;
  }

  pthread_mutex_lock(&acc->mutex);

  PageSlot *slot = &acc->slots[page->page_index];

  // Check if slot is already used
  if (slot->state != PAGE_SLOT_EMPTY) {
    pthread_mutex_unlock(&acc->mutex);
    verboseLog(VERBOSE_NORMAL,
               "Page accumulator: page %d already submitted (state=%d)\n",
               page->page_index + 1, slot->state);
    return false;
  }

  // Copy page info and take ownership of data
  slot->page = *page;
  slot->state = PAGE_SLOT_PENDING;

  verboseLog(VERBOSE_MORE, "Page accumulator: received page %d/%d\n",
             page->page_index + 1, acc->total_pages);

  // Signal writer thread if this is the next page needed
  if (page->page_index == acc->next_write_index) {
    pthread_cond_signal(&acc->page_ready);
  }

  pthread_mutex_unlock(&acc->mutex);

  return true;
}

void pdf_page_accumulator_mark_failed(PdfPageAccumulator *acc, int page_index) {
  if (!acc || page_index < 0 || page_index >= acc->total_pages) {
    return;
  }

  pthread_mutex_lock(&acc->mutex);

  PageSlot *slot = &acc->slots[page_index];
  if (slot->state == PAGE_SLOT_EMPTY) {
    slot->state = PAGE_SLOT_FAILED;
    verboseLog(VERBOSE_MORE, "Page accumulator: marked page %d as failed\n",
               page_index + 1);

    // Signal writer thread if this is the next page needed
    if (page_index == acc->next_write_index) {
      pthread_cond_signal(&acc->page_ready);
    }
  }

  pthread_mutex_unlock(&acc->mutex);
}

bool pdf_page_accumulator_wait(PdfPageAccumulator *acc) {
  if (!acc) {
    return false;
  }

  // Signal that no more pages are coming
  atomic_store(&acc->shutdown, true);
  pthread_mutex_lock(&acc->mutex);
  pthread_cond_signal(&acc->page_ready);
  pthread_mutex_unlock(&acc->mutex);

  // Wait for writer thread to finish
  if (acc->writer_running) {
    pthread_join(acc->writer_thread, NULL);
    acc->writer_running = false;
  }

  int written = atomic_load(&acc->pages_written);
  int failed = atomic_load(&acc->pages_failed);

  verboseLog(VERBOSE_NORMAL, "Page accumulator: %d written, %d failed\n",
             written, failed);

  return (failed == 0);
}

int pdf_page_accumulator_pages_written(const PdfPageAccumulator *acc) {
  return acc ? atomic_load(&acc->pages_written) : 0;
}

int pdf_page_accumulator_pages_failed(const PdfPageAccumulator *acc) {
  return acc ? atomic_load(&acc->pages_failed) : 0;
}
