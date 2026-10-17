/**
 * MLFont Web Loader & HTML5 Performance API Latency Telemetry Library (mlfont.js)
 *
 * Open-source reproduction of Google's machine learning-driven font slicing
 * technology (https://x.com/googlefonts/status/900019727732531200,
 * https://www.tdcommons.org/dpubs_series/906/).
 *
 * Features:
 * 1. Drop-in Sliced Webfont Loader (`MLFont.load` / `data-mlfont` attribute):
 *    Dynamically loads MLFont `@font-face` `unicode-range` stylesheets from
 *    Hugging Face / jsDelivr / self-hosted CDNs so the browser only downloads
 *    the `.woff2` font slices required by the characters on the current page.
 *
 * 2. HTML5 Performance API Latency Metrics (`MLFont.getMetrics` / `MLFont.printReport`):
 *    Accesses the browser Resource Timing & Navigation Timing APIs to measure:
 *      - Download time metrics of individual `.woff2` font slices (slice index,
 *        duration, TTFB, DNS/TCP connect time, transfer size, cache hit status)
 *      - Download time of the sliced `@font-face` CSS stylesheet
 *      - Page onload (`loadEventEnd`) and First Contentful Paint (`first-contentful-paint`)
 *    No user-specific or personally identifiable data (PII) is collected.
 *
 * Quickstart (in <head>):
 *   <script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
 *           data-mlfont="notosanssc"
 *           async></script>
 *
 * License: Apache-2.0
 */
(function(global) {
  'use strict';

  var HF_BASE_URL = 'https://huggingface.co/wq2012';

  /**
   * Registry of pretrained MLFont sliced Noto font bundles on Hugging Face.
   */
  var PRETRAINED_FONTS = {
    'notosanssc': {
      repo: 'mlfont-notosanssc',
      family: 'Noto Sans SC Sliced',
      legacyFamily: 'NotoSansSC-Regular Sliced',
      slices: 102,
      codepoints: 30759
    },
    'sc': {
      repo: 'mlfont-notosanssc',
      family: 'Noto Sans SC Sliced',
      legacyFamily: 'NotoSansSC-Regular Sliced',
      slices: 102,
      codepoints: 30759
    },
    'zh-cn': {
      repo: 'mlfont-notosanssc',
      family: 'Noto Sans SC Sliced',
      legacyFamily: 'NotoSansSC-Regular Sliced',
      slices: 102,
      codepoints: 30759
    },
    'zh-hans': {
      repo: 'mlfont-notosanssc',
      family: 'Noto Sans SC Sliced',
      legacyFamily: 'NotoSansSC-Regular Sliced',
      slices: 102,
      codepoints: 30759
    },
    'notosanstc': {
      repo: 'mlfont-notosanstc',
      family: 'Noto Sans TC Sliced',
      legacyFamily: 'NotoSansTC-Regular Sliced',
      slices: 82,
      codepoints: 20637
    },
    'tc': {
      repo: 'mlfont-notosanstc',
      family: 'Noto Sans TC Sliced',
      legacyFamily: 'NotoSansTC-Regular Sliced',
      slices: 82,
      codepoints: 20637
    },
    'zh-tw': {
      repo: 'mlfont-notosanstc',
      family: 'Noto Sans TC Sliced',
      legacyFamily: 'NotoSansTC-Regular Sliced',
      slices: 82,
      codepoints: 20637
    },
    'zh-hant': {
      repo: 'mlfont-notosanstc',
      family: 'Noto Sans TC Sliced',
      legacyFamily: 'NotoSansTC-Regular Sliced',
      slices: 82,
      codepoints: 20637
    },
    'notosansjp': {
      repo: 'mlfont-notosansjp',
      family: 'Noto Sans JP Sliced',
      legacyFamily: 'NotoSansJP-Regular Sliced',
      slices: 62,
      codepoints: 16663
    },
    'jp': {
      repo: 'mlfont-notosansjp',
      family: 'Noto Sans JP Sliced',
      legacyFamily: 'NotoSansJP-Regular Sliced',
      slices: 62,
      codepoints: 16663
    },
    'ja': {
      repo: 'mlfont-notosansjp',
      family: 'Noto Sans JP Sliced',
      legacyFamily: 'NotoSansJP-Regular Sliced',
      slices: 62,
      codepoints: 16663
    },
    'notosanskr': {
      repo: 'mlfont-notosanskr',
      family: 'Noto Sans KR Sliced',
      legacyFamily: 'NotoSansKR-Regular Sliced',
      slices: 52,
      codepoints: 22853
    },
    'kr': {
      repo: 'mlfont-notosanskr',
      family: 'Noto Sans KR Sliced',
      legacyFamily: 'NotoSansKR-Regular Sliced',
      slices: 52,
      codepoints: 22853
    },
    'ko': {
      repo: 'mlfont-notosanskr',
      family: 'Noto Sans KR Sliced',
      legacyFamily: 'NotoSansKR-Regular Sliced',
      slices: 52,
      codepoints: 22853
    },
    'notosans-lgc': {
      repo: 'mlfont-notosans-lgc',
      family: 'Noto Sans Sliced',
      legacyFamily: 'NotoSans-Regular Sliced',
      slices: 20,
      codepoints: 2297
    },
    'notosans_lgc': {
      repo: 'mlfont-notosans-lgc',
      family: 'Noto Sans Sliced',
      legacyFamily: 'NotoSans-Regular Sliced',
      slices: 20,
      codepoints: 2297
    },
    'notosans': {
      repo: 'mlfont-notosans-lgc',
      family: 'Noto Sans Sliced',
      legacyFamily: 'NotoSans-Regular Sliced',
      slices: 20,
      codepoints: 2297
    },
    'lgc': {
      repo: 'mlfont-notosans-lgc',
      family: 'Noto Sans Sliced',
      legacyFamily: 'NotoSans-Regular Sliced',
      slices: 20,
      codepoints: 2297
    }
  };

  var FontResourceTypes = {
    SLICE_STATIC_CONTENT: 'ef',
    SLICE_CSS_API: 'ec'
  };

  function generateUUID() {
    if (global.crypto && typeof global.crypto.randomUUID === 'function') {
      return global.crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = (Math.random() * 16) | 0;
      var v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  var requestUUID = generateUUID();
  var reportsSent = 0;
  var seenEntries = {};

  var REPORTS = {
    PING: {'t': 'p', 'uuid': requestUUID},
    NO_REPORTS: {'t': 'nr', 'uuid': requestUUID},
    NO_RESOURCE_TIMING: {'t': 'n', 'uuid': requestUUID},
    LOAD_TIMES: {'t': 'lt', 'uuid': requestUUID}
  };

  var telemetryStore = {
    images: {},
    records: []
  };

  function runOnce(name, fn) {
    if (!seenEntries[name]) {
      seenEntries[name] = true;
      return fn();
    }
    return false;
  }

  function getUrl(timings, host) {
    var separator = host.indexOf('?') === -1 ? '?' : '&';
    var url = host;
    for (var key in timings) {
      if (Object.prototype.hasOwnProperty.call(timings, key)) {
        url += separator + encodeURIComponent(key) + '=' +
               encodeURIComponent(String(timings[key]));
        separator = '&';
      }
    }
    return url;
  }

  function sendBeaconOrPixel(timings, host) {
    if (!host || typeof document === 'undefined') {
      return;
    }
    var targetUrl = getUrl(timings, host);
    if (global.navigator && typeof global.navigator.sendBeacon === 'function') {
      try {
        if (global.navigator.sendBeacon(targetUrl)) {
          return;
        }
      } catch (e) {
        // Fall through to <img> beacon fallback
      }
    }
    var id = String(Date.now()) + '_' + Math.random().toString(36).slice(2);
    var img = document.createElement('img');
    img.src = targetUrl;
    telemetryStore.images[id] = img;
    img.onload = img.onerror = function() {
      delete telemetryStore.images[id];
    };
  }

  function createReport(timings, host) {
    if (timings['t'] === FontResourceTypes.SLICE_STATIC_CONTENT ||
        timings['t'] === FontResourceTypes.SLICE_CSS_API) {
      reportsSent++;
    }
    telemetryStore.records.push({
      'timings': timings,
      'host': host || null
    });
  }

  function sendOutReports() {
    var records = telemetryStore.records;
    for (var i = 0; i < records.length; i++) {
      var record = records[i];
      if (record['host'] && !record['sent']) {
        record['sent'] = true;
        sendBeaconOrPixel(record['timings'], record['host']);
      }
    }
  }

  /**
   * Parses a Resource Timing entry for a font slice or sliced CSS file.
   */
  function parseFontResourceEntry(type, entry, interval) {
    var timings = {};
    var cleanUrl = entry.name.split('?')[0];
    var parts = cleanUrl.split('/');
    var fileName = parts[parts.length - 1] || '';

    if (type === FontResourceTypes.SLICE_CSS_API) {
      timings['n'] = fileName.replace(/\.css$/i, '');
    } else if (type === FontResourceTypes.SLICE_STATIC_CONTENT) {
      // Match `<Stem>_slice_<NNN>.<ext>`
      var osMatch = fileName.match(/^(.+?)_slice_(\d+)\.(woff2|woff|otf|ttf)$/i);
      // Match `<Family>-<Weight>.<N>.<ext>`
      var altMatch = fileName.match(/^(.+?)-([A-Za-z]+)\.(\d+)\.(woff2|woff|otf|ttf)$/i);

      if (osMatch) {
        timings['n'] = osMatch[1];
        timings['v'] = 'os-v1';
        timings['w'] = osMatch[1].indexOf('-') !== -1 ?
            osMatch[1].split('-').pop().toLowerCase() : 'regular';
        timings['sn'] = String(parseInt(osMatch[2], 10));
      } else if (altMatch) {
        timings['n'] = parts.length >= 3 ? parts[parts.length - 3] : altMatch[1];
        timings['v'] = parts.length >= 2 ? parts[parts.length - 2] : 'v2';
        timings['w'] = altMatch[2].toLowerCase();
        timings['sn'] = altMatch[3];
      } else {
        timings['n'] = fileName;
        timings['sn'] = '-1';
      }
    }

    timings['t'] = type;
    timings['s'] = Math.round(entry.startTime || 0);
    timings['ds'] = Math.round(entry.domainLookupStart || 0);
    timings['de'] = Math.round(entry.domainLookupEnd || 0);
    timings['cs'] = Math.round(entry.connectStart || 0);
    timings['ce'] = Math.round(entry.connectEnd || 0);
    timings['rq'] = Math.round(entry.requestStart || 0);
    timings['rs'] = Math.round(entry.responseStart || 0);
    timings['re'] = Math.round(entry.responseEnd || 0);
    timings['d'] = Math.round(entry.duration || 0);
    timings['sz'] = typeof entry.transferSize === 'number' ? entry.transferSize : 0;
    timings['cached'] = (typeof entry.transferSize === 'number' &&
                         entry.transferSize === 0 &&
                         entry.decodedBodySize > 0) ? 1 : 0;
    timings['i'] = interval;
    timings['uuid'] = requestUUID;
    return timings;
  }

  function isFontSliceEntry(entry) {
    var name = entry.name || '';
    var clean = name.split('?')[0];
    return (
      /_slice_\d+\.(woff2|woff|otf|ttf)$/i.test(clean) ||
      /\.\d+\.(woff2|woff|otf|ttf)$/i.test(clean) ||
      (name.indexOf('mlfont-') !== -1 && /\.(woff2|woff|otf|ttf)/i.test(name))
    );
  }

  function isSlicedCssEntry(entry) {
    var name = entry.name || '';
    var clean = name.split('?')[0];
    return (
      /font_sliced(_cdn)?\.css$/i.test(clean) ||
      /_sliced\.css$/i.test(clean)
    );
  }

  function scanResourceEntries(host, interval) {
    if (!global.performance || typeof global.performance.getEntriesByType !== 'function') {
      return;
    }
    var entries = global.performance.getEntriesByType('resource') || [];
    for (var i = 0; i < entries.length; i++) {
      var entry = entries[i];
      if (!entry.responseEnd) {
        continue;
      }
      if (isFontSliceEntry(entry)) {
        (function(e) {
          runOnce('slice:' + e.name, function() {
            createReport(
              parseFontResourceEntry(
                FontResourceTypes.SLICE_STATIC_CONTENT, e, interval
              ),
              host
            );
            return true;
          });
        })(entry);
      } else if (isSlicedCssEntry(entry)) {
        (function(e) {
          runOnce('css:' + e.name, function() {
            createReport(
              parseFontResourceEntry(
                FontResourceTypes.SLICE_CSS_API, e, interval
              ),
              host
            );
            return true;
          });
        })(entry);
      }
    }
  }

  function recordLoadTimes(host) {
    if (!global.performance) {
      return;
    }
    var loadTimes = {'t': REPORTS.LOAD_TIMES['t'], 'uuid': requestUUID};
    var added = false;

    if (typeof global.performance.getEntriesByType === 'function') {
      var navEntries = global.performance.getEntriesByType('navigation');
      if (navEntries && navEntries.length > 0) {
        var nav = navEntries[0];
        if (nav.loadEventEnd > 0) {
          added = runOnce('ld', function() {
            loadTimes['ld'] = Math.round(nav.loadEventEnd);
            loadTimes['dcl'] = Math.round(nav.domContentLoadedEventEnd || 0);
            return true;
          }) || added;
        }
      }
      var paintEntries = global.performance.getEntriesByType('paint') || [];
      for (var i = 0; i < paintEntries.length; i++) {
        var p = paintEntries[i];
        if (p.name === 'first-paint') {
          added = runOnce('fp', function() {
            loadTimes['fp'] = Math.round(p.startTime);
            return true;
          }) || added;
        } else if (p.name === 'first-contentful-paint') {
          added = runOnce('fcp', function() {
            loadTimes['fcp'] = Math.round(p.startTime);
            return true;
          }) || added;
        }
      }
    } else if (global.performance.timing && global.performance.timing.navigationStart) {
      var t = global.performance.timing;
      if (t.loadEventEnd > 0) {
        added = runOnce('ld', function() {
          loadTimes['ld'] = t.loadEventEnd - t.navigationStart;
          return true;
        }) || added;
      }
    }

    if (added) {
      createReport(loadTimes, host);
    }
  }

  /**
   * Returns a structured summary of all downloaded MLFont slices, CSS stylesheets,
   * and page load metrics for the current browsing session.
   */
  function getMetrics() {
    scanResourceEntries(null, -1);
    recordLoadTimes(null);

    var slices = [];
    var stylesheets = [];
    var pageLoad = {};
    var totalSliceBytes = 0;
    var cachedSlicesCount = 0;
    var maxSliceDurationMs = 0;
    var maxSliceResponseEnd = 0;
    var minSliceStartTime = Infinity;

    var records = telemetryStore.records;
    for (var i = 0; i < records.length; i++) {
      var rec = records[i]['timings'];
      if (rec['t'] === FontResourceTypes.SLICE_STATIC_CONTENT) {
        var sliceInfo = {
          font: rec['n'],
          sliceIndex: parseInt(rec['sn'], 10),
          weight: rec['w'] || 'regular',
          startTimeMs: Number(rec['s']),
          ttfbMs: Math.max(0, Number(rec['rs']) - Number(rec['rq'])),
          durationMs: Number(rec['d']),
          responseEndMs: Number(rec['re']),
          transferBytes: Number(rec['sz'] || 0),
          cached: Boolean(rec['cached'])
        };
        slices.push(sliceInfo);
        totalSliceBytes += sliceInfo.transferBytes;
        if (sliceInfo.cached) {
          cachedSlicesCount++;
        }
        if (sliceInfo.durationMs > maxSliceDurationMs) {
          maxSliceDurationMs = sliceInfo.durationMs;
        }
        if (sliceInfo.responseEndMs > maxSliceResponseEnd) {
          maxSliceResponseEnd = sliceInfo.responseEndMs;
        }
        if (sliceInfo.startTimeMs < minSliceStartTime) {
          minSliceStartTime = sliceInfo.startTimeMs;
        }
      } else if (rec['t'] === FontResourceTypes.SLICE_CSS_API) {
        stylesheets.push({
          stylesheet: rec['n'],
          startTimeMs: Number(rec['s']),
          durationMs: Number(rec['d']),
          transferBytes: Number(rec['sz'] || 0)
        });
      } else if (rec['t'] === REPORTS.LOAD_TIMES['t']) {
        if (rec['ld'] !== undefined) pageLoad.loadEventEndMs = rec['ld'];
        if (rec['dcl'] !== undefined) pageLoad.domContentLoadedMs = rec['dcl'];
        if (rec['fp'] !== undefined) pageLoad.firstPaintMs = rec['fp'];
        if (rec['fcp'] !== undefined) pageLoad.firstContentfulPaintMs = rec['fcp'];
      }
    }

    slices.sort(function(a, b) {
      return a.sliceIndex - b.sliceIndex;
    });

    var wallClockMs = slices.length > 0 ?
        Math.max(0, maxSliceResponseEnd - minSliceStartTime) : 0;

    return {
      uuid: requestUUID,
      numSlicesDownloaded: slices.length,
      totalSliceTransferBytes: totalSliceBytes,
      wallClockFontDownloadMs: wallClockMs,
      summary: {
        totalSlicesFetched: slices.length,
        cachedSlicesCount: cachedSlicesCount,
        totalTransferBytes: totalSliceBytes,
        maxSliceDurationMs: maxSliceDurationMs,
        wallClockFontDownloadMs: wallClockMs
      },
      slices: slices,
      stylesheets: stylesheets,
      pageLoad: pageLoad
    };
  }

  /**
   * Prints a formatted table of downloaded font slices and latency metrics to DevTools console.
   */
  function printReport() {
    var metrics = getMetrics();
    if (typeof console !== 'undefined' && console.log) {
      console.log(
        '[MLFont] Slices loaded: ' + metrics.numSlicesDownloaded +
        ' | Transfer: ' + (metrics.totalSliceTransferBytes / 1024).toFixed(1) + ' KB' +
        ' | Wall-clock font load: ' + metrics.wallClockFontDownloadMs + ' ms'
      );
      if (metrics.slices.length > 0 && typeof console.table === 'function') {
        console.table(metrics.slices);
      }
    }
    return metrics;
  }

  /**
   * Dynamically loads a sliced font stylesheet by preset name ('notosanssc', 'notosanstc',
   * 'notosansjp', 'notosanskr', 'notosans-lgc') or custom CSS URL, and injects the
   * `@font-face` rules into `document.head`.
   *
   * @param {string} fontOrUrl Preset key or URL to `font_sliced.css` / `font_sliced_cdn.css`.
   * @param {Object=} options Optional `{ baseUrl, fontDisplay }` overrides.
   * @return {!Promise<!HTMLStyleElement>} Promise resolving to the injected `<style>` element.
   */
  function loadSlicedFont(fontOrUrl, options) {
    options = options || {};
    var key = String(fontOrUrl || '').trim().toLowerCase();
    var preset = PRETRAINED_FONTS[key];
    var cssUrl = fontOrUrl;
    var woff2BaseUrl = options.baseUrl || '';

    if (preset) {
      var repoBase = (options.hfBaseUrl || HF_BASE_URL) + '/' + preset.repo + '/resolve/main';
      cssUrl = repoBase + '/font_sliced_cdn.css';
      woff2BaseUrl = repoBase + '/woff2/';
    } else if (!woff2BaseUrl && cssUrl.indexOf('/') !== -1) {
      woff2BaseUrl = cssUrl.substring(0, cssUrl.lastIndexOf('/') + 1);
    }

    if (typeof document === 'undefined' || typeof fetch !== 'function') {
      return Promise.reject(new Error('MLFont.load requires a browser environment with fetch().'));
    }

    var styleId = 'mlfont-style-' + key.replace(/[^a-z0-9_-]/g, '-');
    var existing = document.getElementById(styleId);
    if (existing) {
      return Promise.resolve(existing);
    }

    return fetch(cssUrl)
      .then(function(res) {
        if (!res.ok && preset) {
          // Fallback to font_sliced.css if font_sliced_cdn.css is not yet present
          var fallbackUrl = (options.hfBaseUrl || HF_BASE_URL) + '/' +
                            preset.repo + '/resolve/main/font_sliced.css';
          return fetch(fallbackUrl);
        }
        return res;
      })
      .then(function(res) {
        if (!res.ok) {
          throw new Error('Failed to fetch MLFont CSS from ' + cssUrl + ' (' + res.status + ')');
        }
        return res.text();
      })
      .then(function(cssText) {
        // Rewrite relative `url(foo.woff2)` or `url(woff2/foo.woff2)` to absolute CDN URLs
        var rewritten = cssText.replace(
          /url\((['"]?)(?!https?:\/\/|\/\/|data:)(?:woff2\/)?([^)'"]+\.woff2)\1\)/gi,
          function(_match, _quote, file) {
            return "url('" + woff2BaseUrl + file + "')";
          }
        );

        // If preset has both human-friendly family ('Noto Sans SC Sliced') and stem family
        // ('NotoSansSC-Regular Sliced'), ensure both family names are registered.
        if (preset &&
            rewritten.indexOf("font-family: '" + preset.family + "'") === -1 &&
            rewritten.indexOf("font-family: '" + preset.legacyFamily + "'") !== -1) {
          var aliased = rewritten.replace(
            new RegExp("font-family:\\s*'" + preset.legacyFamily + "'", 'g'),
            "font-family: '" + preset.family + "'"
          );
          rewritten = aliased + '\n' + rewritten;
        }

        var styleEl = document.createElement('style');
        styleEl.id = styleId;
        styleEl.setAttribute('data-mlfont-source', cssUrl);
        styleEl.textContent = rewritten;
        (document.head || document.documentElement).appendChild(styleEl);
        return styleEl;
      });
  }

  /**
   * Initializes the periodic HTML5 Performance API resource timing collector.
   */
  function setupTelemetry(hostOrOptions) {
    var host = typeof hostOrOptions === 'string' ?
        hostOrOptions : (hostOrOptions && hostOrOptions.endpoint) || null;
    var debug = typeof hostOrOptions === 'object' && Boolean(hostOrOptions && hostOrOptions.debug);

    if (global.performance && typeof global.performance.getEntriesByType === 'function') {
      setTimeout(function() {
        if (host) sendBeaconOrPixel(REPORTS.PING, host);
        scanResourceEntries(host, 0);
      }, 1);
      setTimeout(function() {
        scanResourceEntries(host, 2);
        recordLoadTimes(host);
      }, 2000);
      setTimeout(function() {
        scanResourceEntries(host, 5);
        recordLoadTimes(host);
        if (reportsSent === 0 && host) {
          sendBeaconOrPixel(REPORTS.NO_REPORTS, host);
        }
        if (debug) {
          printReport();
        }
      }, 5000);
      setTimeout(function() {
        scanResourceEntries(host, 10);
        recordLoadTimes(host);
        sendOutReports();
      }, 10000);
    } else if (host) {
      sendBeaconOrPixel(REPORTS.NO_RESOURCE_TIMING, host);
    }
  }

  var MLFontAPI = {
    version: '0.1.0',
    PRETRAINED_FONTS: PRETRAINED_FONTS,
    load: loadSlicedFont,
    setup: setupTelemetry,
    getMetrics: getMetrics,
    printReport: printReport,
    records: telemetryStore.records
  };

  global['MLFont'] = MLFontAPI;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLFontAPI;
  }

  // Auto-initialize when loaded via `<script src=".../mlfont.js" ...>` in a browser
  if (typeof document !== 'undefined') {
    var currentScript = document.currentScript;
    if (!currentScript) {
      var scripts = document.getElementsByTagName('script');
      for (var i = scripts.length - 1; i >= 0; i--) {
        if (scripts[i].src && scripts[i].src.indexOf('mlfont.js') !== -1) {
          currentScript = scripts[i];
          break;
        }
      }
    }

    var fontAttr = currentScript && currentScript.getAttribute('data-mlfont');
    var endpointAttr = currentScript && currentScript.getAttribute('data-endpoint');
    var debugAttr = currentScript && currentScript.getAttribute('data-debug') === 'true';

    if (fontAttr) {
      var fontList = fontAttr.split(',');
      for (var j = 0; j < fontList.length; j++) {
        (function(fontName) {
          if (fontName) {
            loadSlicedFont(fontName).catch(function(err) {
              if (typeof console !== 'undefined' && console.warn) {
                console.warn('[MLFont] Failed to auto-load ' + fontName + ':', err);
              }
            });
          }
        })(fontList[j].trim());
      }
    }

    setupTelemetry({endpoint: endpointAttr || null, debug: debugAttr});
  }
})(typeof window !== 'undefined' ? window : globalThis);
