/* ────────────────────────────────────────────────────────────
   AI Видео Транскрибатор · app.js
   ──────────────────────────────────────────────────────────── */

class VideoTranscriber {
  constructor() {
    this.currentTaskId  = null;
    this.eventSource    = null;
    this.apiBase        = '/api';
    this.currentLang    = 'ru';
    this.currentTheme   = localStorage.getItem('vt_theme') || 'dark';

    /* Имитация прогресса */
    this.sp = {
      enabled: false, current: 0, target: 15,
      lastServer: 0, interval: null, startTime: null, stage: 'preparing'
    };

    this.i18n = {
      en: {
        title:                   'AI Video Transcriber',
        subtitle:                'Supports automatic transcription and AI summary for 30+ platforms',
        video_url_placeholder:   'Paste YouTube, Tiktok, Bilibili or other platform video URLs...',
        start_transcription:     'Transcribe',
        ai_settings:             'AI Settings',
        model_base_url:          'Model API Base URL',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            'Fetch',
        model_select:            'Model',
        model_default:           '— use server default —',
        summary_language:        'Summary Language',
        transcription_language:  'Transcription Language',
        transcription_auto:      'Auto',
        processing_progress:     'Processing',
        preparing:               'Preparing…',
        transcript_text:         'Transcript',
        intelligent_summary:     'AI Summary',
        translation:             'Translation',
        download_transcript:     'Transcript',
        download_translation:    'Translation',
        download_summary:        'Summary',
        empty_hint:              'Paste a video URL or drop a file above and let AI do the heavy lifting.',
        footer_text:             ' ',
        processing:              'Processing…',
        downloading_video:       'Downloading audio…',
        parsing_video:           'Parsing video info…',
        transcribing_audio:      'Transcribing audio…',
        optimizing_transcript:   'Optimizing transcript…',
        generating_summary:      'Generating summary…',
        detecting_subtitles:     'Detecting subtitles…',
        subtitle_found:          'Subtitles found! Processing text…',
        no_subtitle:             'No subtitles found, downloading audio…',
        mode_subtitle:           '⚡ Subtitle',
        mode_whisper:            '🎙 Whisper',
        completed:               'Done!',
        error_invalid_url:       'Please enter a valid video URL',
        error_processing_failed: 'Processing failed: ',
        error_no_download:       'No file available for download',
        error_download_failed:   'Download failed: ',
        fetching_models:         'Fetching models…',
        models_loaded:           (n) => `${n} models loaded`,
        models_error:            'Failed to fetch models',
        upload_or:               'or drop your files',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        'Upload files',
        error_upload_type:       'Unsupported file type',
        error_upload_empty:      'File is empty',
        error_upload_size:       (mb) => `File exceeds ${mb} MB limit`,
      },
      zh: {
        title:                   'AI 视频转录器',
        subtitle:                '粘贴 YouTube、TikTok 或任意公开视频链接，获取转录文本和 AI 摘要。',
        video_url_placeholder:   '请输入视频链接…',
        start_transcription:     '开始转录',
        ai_settings:             'AI 设置',
        model_base_url:          'Model API 地址',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            '获取',
        model_select:            '模型',
        model_default:           '— 使用服务器默认 —',
        summary_language:        '摘要语言',
        transcription_language:  '转录语言',
        transcription_auto:      '自动',
        processing_progress:     '处理进度',
        preparing:               '准备中…',
        transcript_text:         '转录文本',
        intelligent_summary:     '智能摘要',
        translation:             '翻译',
        download_transcript:     '转录',
        download_translation:    '翻译',
        download_summary:        '摘要',
        empty_hint:              '在上方粘贴视频链接或拖放文件，让 AI 来处理一切。',
        footer_text:             '本工具是 <a href="https://sipsip.ai" target="_blank" style="color:var(--accent-text);text-decoration:none;">sipsip.ai</a> 的一部分 — 提取任何内容要点并构建你自己的知识库。',
        processing:              '处理中…',
        downloading_video:       '正在下载音频…',
        parsing_video:           '正在解析视频信息…',
        transcribing_audio:      '正在转录音频…',
        optimizing_transcript:   '正在优化转录文本…',
        generating_summary:      '正在生成摘要…',
        detecting_subtitles:     '正在检测字幕…',
        subtitle_found:          '字幕获取成功！正在处理文本…',
        no_subtitle:             '未找到字幕，正在下载音频…',
        mode_subtitle:           '⚡ 字幕模式',
        mode_whisper:            '🎙 Whisper 模式',
        completed:               '处理完成！',
        error_invalid_url:       '请输入有效的视频链接',
        error_processing_failed: '处理失败：',
        error_no_download:       '没有可下载的文件',
        error_download_failed:   '下载失败：',
        fetching_models:         '正在获取模型列表…',
        models_loaded:           (n) => `已加载 ${n} 个模型`,
        models_error:            '获取模型失败',
        upload_or:               '或拖放文件到此处',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        '上传文件',
        error_upload_type:       '不支持的文件类型',
        error_upload_empty:      '文件为空',
        error_upload_size:       (mb) => `文件超过 ${mb} MB 限制`,
      },
      ru: {
        title:                   'AI Видео Транскрибатор',
        subtitle:                'Автоматическая транскрипция и ИИ-резюме для 30+ платформ',
        video_url_placeholder:   'Вставьте ссылку на видео с YouTube, TikTok, Bilibili или других платформ...',
        start_transcription:     'Транскрибировать',
        ai_settings:             'Настройки AI',
        model_base_url:          'API-адрес модели',
        model_base_url_placeholder: 'https://openrouter.ai/api/v1',
        api_key:                 'API Key',
        api_key_placeholder:     'sk-...',
        fetch_models:            'Получить',
        model_select:            'Модель',
        model_default:           '— использовать серверную по умолчанию —',
        summary_language:        'Язык резюме',
        transcription_language:  'Язык транскрипции',
        transcription_auto:      'Авто',
        processing_progress:     'Обработка',
        preparing:               'Подготовка…',
        transcript_text:         'Транскрипция',
        intelligent_summary:     'ИИ-резюме',
        translation:             'Перевод',
        download_transcript:     'Транскрипция',
        download_translation:    'Перевод',
        download_summary:        'Резюме',
        empty_hint:              'Вставьте ссылку на видео или загрузите файл, и ИИ сделает всю работу.',
        footer_text:             ' ',
        processing:              'Обработка…',
        downloading_video:       'Загрузка аудио…',
        parsing_video:           'Анализ информации о видео…',
        transcribing_audio:      'Транскрипция аудио…',
        optimizing_transcript:   'Оптимизация транскрипции…',
        generating_summary:      'Создание резюме…',
        detecting_subtitles:     'Поиск субтитров…',
        subtitle_found:          'Субтитры найдены! Обработка текста…',
        no_subtitle:             'Субтитры не найдены, загрузка аудио…',
        mode_subtitle:           '⚡ Субтитры',
        mode_whisper:            '🎙 Whisper',
        completed:               'Готово!',
        error_invalid_url:       'Пожалуйста, введите корректную ссылку на видео',
        error_processing_failed: 'Ошибка обработки: ',
        error_no_download:       'Нет файлов для скачивания',
        error_download_failed:   'Ошибка скачивания: ',
        fetching_models:         'Получение списка моделей…',
        models_loaded:           (n) => `Загружено ${n} моделей`,
        models_error:            'Ошибка получения моделей',
        upload_or:               'или перетащите файлы',
        upload_formats:          '.mp3 · .mp4 · .wav · .m4a · .webm · .mkv · .ogg · .flac',
        upload_files_btn:        'Загрузить файлы',
        error_upload_type:       'Неподдерживаемый тип файла',
        error_upload_empty:      'Файл пуст',
        error_upload_size:       (mb) => `Файл превышает лимит ${mb} МБ`,
      }
    };

    this.systemVersion = 'unknown';
    this.systemInfo = null;

    this._initElements();
    this._bindEvents();
    this._loadSettings();
    this._applyTheme(this.currentTheme);
    this._switchLang(this.currentLang);

    // Загрузка информации о системе после инициализации
    setTimeout(() => this._loadSystemInfo(), 500);
  }

  /* ── Элементы ─────────────────────────────────────────── */
  _initElements() {
    this.form               = document.getElementById('videoForm');
    this.videoUrlInput      = document.getElementById('videoUrl');
    this.submitBtn          = document.getElementById('submitBtn');
    this.summaryLangSel     = document.getElementById('summaryLanguage');
    this.transcriptionLangSel = document.getElementById('transcriptionLanguage');
    this.langToggle         = document.getElementById('langToggle');
    this.langText           = document.getElementById('langText');
    this.themeToggle        = document.getElementById('themeToggle');
    this.errorBanner        = document.getElementById('errorBanner');
    this.errorMsg           = document.getElementById('errorMsg');
    this.emptyState         = document.getElementById('emptyState');
    this.progressPanel      = document.getElementById('progressPanel');
    this.modeBadge          = document.getElementById('modeBadge');
    this.progressStatus     = document.getElementById('progressStatus');
    this.progressFill       = document.getElementById('progressFill');
    this.progressMessage    = document.getElementById('progressMessage');
    this.resultsPanel       = document.getElementById('resultsPanel');
    this.scriptContent      = document.getElementById('scriptContent');
    this.summaryContent     = document.getElementById('summaryContent');
    this.translationContent = document.getElementById('translationContent');
    this.dlScript           = document.getElementById('downloadScript');
    this.dlTranslation      = document.getElementById('downloadTranslation');
    this.dlSummary          = document.getElementById('downloadSummary');
    this.translationTabBtn  = document.getElementById('translationTabBtn');
    this.tabBtns            = document.querySelectorAll('.tab-btn');
    this.tabPanes           = document.querySelectorAll('.tab-pane');
    // Настройки
    this.settingsToggle     = document.getElementById('settingsToggle');
    this.settingsBody       = document.getElementById('settingsBody');
    this.modelBaseUrl       = document.getElementById('modelBaseUrl');
    this.apiKeyInput        = document.getElementById('apiKeyInput');
    this.fetchModelsBtn     = document.getElementById('fetchModelsBtn');
    this.fetchStatus        = document.getElementById('fetchStatus');
    this.modelSelect        = document.getElementById('modelSelect');
    this.fetchIcon          = document.getElementById('fetchIcon');
    this.uploadZone         = document.getElementById('uploadZone');
    this.uploadPickBtn      = document.getElementById('uploadPickBtn');
    this.fileInput          = document.getElementById('fileInput');
    this.uploadMaxMb        = 200;
    this._allowedUploadExts = new Set(['.txt', '.mp3', '.mp4', '.m4a', '.wav', '.webm', '.mkv', '.ogg', '.flac']);
  }

  /* ── Информация о системе ─────────────────────────────── */
  async _loadSystemInfo() {
      try {
          // Получение версии
          const versionResp = await fetch(`${this.apiBase}/version`);
          if (versionResp.ok) {
              const versionData = await versionResp.json();
              this.systemVersion = versionData.version || 'unknown';
          }
          
          // Получение информации о системе
          const infoResp = await fetch(`${this.apiBase}/system-info`);
          if (infoResp.ok) {
              const info = await infoResp.json();
              this.systemInfo = info;
              this._updateFooter();
          }
      } catch (e) {
          console.warn('Не удалось загрузить информацию о системе:', e);
      }
  }

  _updateFooter() {
      const footer = document.querySelector('.footer p[data-i18n="footer_text"]');
      if (!footer) return;
      
      const version = this.systemVersion || 'unknown';
      const device = this.systemInfo?.whisper_device || 'cpu';
      const size = this.systemInfo?.whisper_size || 'unknown';
      const compute = this.systemInfo?.whisper_compute_type || 'int8';
      const cuda = this.systemInfo?.cuda_available ? '✅ GPU' : '💻 CPU';
      const gpuName = this.systemInfo?.cuda_device_name || '';
      
      // Формируем информацию в зависимости от языка
      const lang = this.currentLang;
      let deviceLabel, computeLabel, versionLabel;
      
      if (lang === 'ru') {
          deviceLabel = 'Устройство';
          computeLabel = 'Тип вычислений';
          versionLabel = 'Версия';
      } else if (lang === 'zh') {
          deviceLabel = '设备';
          computeLabel = '计算类型';
          versionLabel = '版本';
      } else {
          deviceLabel = 'Device';
          computeLabel = 'Compute Type';
          versionLabel = 'Version';
      }
      
      // Формируем текст с информацией
      let infoText = `${versionLabel}: ${version} | ${deviceLabel}: ${device} (${compute}) - ${size}`;
      
      if (gpuName && this.systemInfo?.cuda_available) {
          infoText += ` | GPU: ${gpuName}`;
      }
      
      // Добавляем эмодзи статуса
      const statusEmoji = this.systemInfo?.cuda_available ? '🚀' : '💻';
      infoText = `${statusEmoji} ${infoText}`;
      
      // Сохраняем оригинальный footer_text для использования в качестве префикса
      const originalText = this.t('footer_text');
      if (originalText && originalText.trim() && !originalText.includes('|')) {
          // Если есть оригинальный текст, добавляем информацию после него
          footer.innerHTML = `${originalText} <span style="color: var(--text-dim); font-size: 11px; margin-left: 12px;">${infoText}</span>`;
      } else {
          footer.textContent = infoText;
      }
  }

  /* ── События ───────────────────────────────────────────── */
  _bindEvents() {
    this.form.addEventListener('submit', (e) => { e.preventDefault(); this._startTranscription(); });

    this.langToggle.addEventListener('click', () => {
      const langs = ['ru', 'en', 'zh'];
      const currentIdx = langs.indexOf(this.currentLang);
      const nextIdx = (currentIdx + 1) % langs.length;
      this._switchLang(langs[nextIdx]);
    });

    if (this.themeToggle) {
      this.themeToggle.addEventListener('click', () => this._toggleTheme());
    }

    // Переключение настроек
    this.settingsToggle.addEventListener('click', () => {
      const open = this.settingsBody.classList.toggle('open');
      this.settingsToggle.classList.toggle('open', open);
    });

    // Получение списка моделей
    this.fetchModelsBtn.addEventListener('click', () => this._fetchModels());

    // Автоматическое получение моделей при заполнении полей (с задержкой)
    const debouncedFetch = this._debounce(() => {
      if (this.modelBaseUrl.value.trim() && this.apiKeyInput.value.trim()) this._fetchModels();
    }, 900);
    this.modelBaseUrl.addEventListener('input', debouncedFetch);
    this.apiKeyInput.addEventListener('input', debouncedFetch);

    // Сохранение настроек
    [this.modelBaseUrl, this.apiKeyInput, this.modelSelect, this.summaryLangSel, this.transcriptionLangSel].forEach(el => {
      el.addEventListener('change', () => this._saveSettings());
    });

    // Вкладки
    this.tabBtns.forEach(btn => {
      btn.addEventListener('click', () => this._switchTab(btn.dataset.tab));
    });

    // Загрузка файлов
    this.dlScript.addEventListener('click',      () => this._downloadFile('script'));
    this.dlTranslation.addEventListener('click', () => this._downloadFile('translation'));
    this.dlSummary.addEventListener('click',     () => this._downloadFile('summary'));

    if (this.uploadPickBtn && this.fileInput && this.uploadZone) {
      this.uploadPickBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.fileInput.click();
      });
      this.uploadZone.addEventListener('click', (e) => {
        if (e.target === this.uploadPickBtn || this.uploadPickBtn.contains(e.target)) return;
        this.fileInput.click();
      });
      this.fileInput.addEventListener('change', () => {
        const f = this.fileInput.files && this.fileInput.files[0];
        this.fileInput.value = '';
        if (f) this._startFileUpload(f);
      });
      ['dragenter', 'dragover'].forEach((ev) => {
        this.uploadZone.addEventListener(ev, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.uploadZone.classList.add('dragover');
        });
      });
      this.uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        if (!this.uploadZone.contains(e.relatedTarget)) {
          this.uploadZone.classList.remove('dragover');
        }
      });
      this.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        const f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) this._startFileUpload(f);
      });
    }
  }

  /* ── Интернационализация ─────────────────────────────── */
  t(key) { return this.i18n[this.currentLang][key] || this.i18n['en'][key] || key; }

  _switchLang(lang) {
      this.currentLang = lang;
      const langNames = { ru: 'Русский', en: 'English', zh: '中文' };
      this.langText.textContent = langNames[lang] || 'English';
      document.documentElement.lang = lang;
      document.title = this.t('title');

      document.querySelectorAll('[data-i18n]').forEach(el => {
          const v = this.t(el.dataset.i18n);
          if (typeof v === 'string') {
              if (el.dataset.i18n === 'footer_text') {
                  // Для футера используем специальную обработку
                  this._updateFooter();
              } else {
                  el.textContent = v;
              }
          }
      });
      document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
          const v = this.t(el.dataset.i18nPlaceholder);
          if (typeof v === 'string') el.placeholder = v;
      });
  }

  /* ── Тема ───────────────────────────────────────────────── */
  _applyTheme(theme) {
    this.currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('vt_theme', theme);
    
    if (this.themeToggle) {
      const icon = this.themeToggle.querySelector('i');
      if (icon) {
        icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
      }
    }
  }

  _toggleTheme() {
    const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
    this._applyTheme(newTheme);
  }

  /* ── Сохранение настроек ─────────────────────────────── */
  _saveSettings() {
    const s = {
      baseUrl:  this.modelBaseUrl.value,
      apiKey:   this.apiKeyInput.value,
      model:    this.modelSelect.value,
      summaryLang: this.summaryLangSel.value,
      transcriptionLang: this.transcriptionLangSel.value,
      lang:     this.currentLang,
      theme:    this.currentTheme,
    };
    try { localStorage.setItem('vt_settings', JSON.stringify(s)); } catch (_) {}
  }

  _loadSettings() {
    try {
      const raw = localStorage.getItem('vt_settings');
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.baseUrl)          this.modelBaseUrl.value = s.baseUrl;
      if (s.apiKey)           this.apiKeyInput.value  = s.apiKey;
      if (s.summaryLang)      this.summaryLangSel.value = s.summaryLang;
      if (s.transcriptionLang) this.transcriptionLangSel.value = s.transcriptionLang || 'auto';
      if (s.lang)             this.currentLang = s.lang;
      if (s.theme)            this.currentTheme = s.theme;
      this._savedModel = s.model || '';

      if (s.baseUrl || s.apiKey) {
        this.settingsBody.classList.add('open');
        this.settingsToggle.classList.add('open');
        if (s.baseUrl && s.apiKey) {
          setTimeout(() => this._fetchModels(true), 400);
        }
      }
    } catch (_) {}
  }

  /* ── Получение моделей ────────────────────────────────── */
  async _fetchModels(silent = false) {
    const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
    const apiKey  = this.apiKeyInput.value.trim();

    if (!baseUrl || !apiKey) {
      if (!silent) this._setFetchStatus('err', this.t('api_key') + ' & URL required');
      return;
    }

    this.fetchModelsBtn.disabled = true;
    this.fetchIcon.className = 'fas fa-spinner fa-spin';
    if (!silent) this._setFetchStatus('', this.t('fetching_models'));

    try {
      const fd = new FormData();
      fd.append('base_url', baseUrl);
      fd.append('api_key',  apiKey);

      const resp = await fetch(`${this.apiBase}/models`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const models = data.data || data.models || [];

      this.modelSelect.innerHTML = `<option value="">${this.t('model_default')}</option>`;
      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name || m.id;
        this.modelSelect.appendChild(opt);
      });

      if (this._savedModel) {
        this.modelSelect.value = this._savedModel;
        this._savedModel = '';
      }

      this._setFetchStatus('ok', typeof this.t('models_loaded') === 'function'
        ? this.t('models_loaded')(models.length)
        : `${models.length} models`);

    } catch (e) {
      console.warn('Ошибка получения моделей:', e);
      this._setFetchStatus('err', this.t('models_error') + ': ' + e.message);
    } finally {
      this.fetchModelsBtn.disabled = false;
      this.fetchIcon.className = 'fas fa-sync-alt';
    }
  }

  _setFetchStatus(cls, msg) {
    this.fetchStatus.className = 'fetch-status' + (cls ? ` ${cls}` : '');
    this.fetchStatus.textContent = msg;
  }

  /* ── Транскрипция ────────────────────────────────────── */
  async _startTranscription() {
    if (this.submitBtn.disabled) return;

    const url     = this.videoUrlInput.value.trim();
    const sumLang = this.summaryLangSel.value;
    const transLang = this.transcriptionLangSel.value;

    if (!url) { this._showError(this.t('error_invalid_url')); return; }

    this._setLoading(true);
    this._hideError();
    this._showProgress();

    try {
      const fd = new FormData();
      fd.append('url',              url);
      fd.append('summary_language', sumLang);
      fd.append('transcription_language', transLang);

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Ошибка запроса');
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  async _startFileUpload(file) {
    if (this.submitBtn.disabled) return;

    const parts = (file.name || '').split('.');
    const ext = parts.length > 1 ? ('.' + parts.pop().toLowerCase()) : '';
    if (!this._allowedUploadExts.has(ext)) {
      this._showError(this.t('error_upload_type'));
      return;
    }
    if (!file.size) {
      this._showError(this.t('error_upload_empty'));
      return;
    }
    const maxB = this.uploadMaxMb * 1024 * 1024;
    if (file.size > maxB) {
      this._showError(this.t('error_upload_size')(this.uploadMaxMb));
      return;
    }

    this._setLoading(true);
    this._hideError();
    this._showProgress();

    const sumLang = this.summaryLangSel.value;
    const transLang = this.transcriptionLangSel.value;
    try {
      const fd = new FormData();
      fd.append('file', file, file.name);
      fd.append('summary_language', sumLang);
      fd.append('transcription_language', transLang);

      const apiKey  = this.apiKeyInput.value.trim();
      const baseUrl = this.modelBaseUrl.value.trim().replace(/\/$/, '');
      const modelId = this.modelSelect.value;
      if (apiKey)  fd.append('api_key',       apiKey);
      if (baseUrl) fd.append('model_base_url', baseUrl);
      if (modelId) fd.append('model_id',       modelId);

      const resp = await fetch(`${this.apiBase}/process-video`, { method: 'POST', body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        const d = err.detail;
        const msg = typeof d === 'string'
          ? d
          : (Array.isArray(d) && d[0] && (d[0].msg || d[0].message))
            || `HTTP ${resp.status}`;
        throw new Error(msg);
      }

      const data = await resp.json();
      this.currentTaskId = data.task_id;

      this._initSP();
      this._updateProgress(5, this.t('preparing'), true);
      this._startSSE();
      this._saveSettings();

    } catch (err) {
      this._showError(this.t('error_processing_failed') + err.message);
      this._setLoading(false);
      this._hideProgress();
    }
  }

  /* ── SSE ──────────────────────────────────────────────── */
  _startSSE() {
    if (!this.currentTaskId) return;
    this.eventSource = new EventSource(`${this.apiBase}/task-stream/${this.currentTaskId}`);

    this.eventSource.onmessage = (ev) => {
      try {
        const task = JSON.parse(ev.data);
        if (task.type === 'heartbeat') return;

        this._updateProgress(task.progress, task.message, true);

        if (task.status === 'completed') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language);
        } else if (task.status === 'error') {
          this._stopSP(); this._stopSSE(); this._setLoading(false); this._hideProgress();
          this._showError(task.error || 'Ошибка обработки');
        }
      } catch (_) {}
    };

    this.eventSource.onerror = async () => {
      this._stopSSE();
      try {
        if (this.currentTaskId) {
          const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
          if (r.ok) {
            const task = await r.json();
            if (task?.status === 'completed') {
              this._stopSP(); this._setLoading(false); this._hideProgress();
              this._showResults(task.script, task.summary, task.video_title, task.translation, task.detected_language, task.summary_language);
              return;
            }
          }
        }
      } catch (_) {}
      this._showError(this.t('error_processing_failed') + 'SSE отключено');
      this._setLoading(false);
    };
  }

  _stopSSE() {
    if (this.eventSource) { this.eventSource.close(); this.eventSource = null; }
  }

  /* ── Прогресс ─────────────────────────────────────────── */
  _updateProgress(pct, msg, fromServer = false) {
    if (fromServer) {
      this._stopSP();
      this.sp.lastServer = pct;
      this.sp.current    = pct;
      this._renderProgress(pct, msg);
      this._updateStage(pct, msg);
      this._startSP();
    } else {
      this._renderProgress(pct, msg);
    }
  }

  _updateStage(pct, msg) {
    const m = (msg || '').toLowerCase();

    // ── Путь с субтитрами (быстрый) ────────────────────────────
    if (m.includes('субтитры получены') || m.includes('subtitle found') || m.includes('获取成功') || m.includes('字幕获取')) {
      this.sp.stage = 'subtitle_found';
      this.sp.target = 55;
      this._setModeBadge('subtitle');
    }
    // ── Без субтитров → загрузка аудио (медленный) ─────────────
    else if (m.includes('субтитры не найдены') || m.includes('no subtitle') || m.includes('未找到字幕') || m.includes('下载视频音频') || m.includes('下载音频')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('чтение текста') || m.includes('read') && m.includes('text') || m.includes('读取文本')) {
      this.sp.stage = 'parsing';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('преобразование аудио') || m.includes('подготовка к транскрипции') || m.includes('转换音频') || m.includes('准备转录')) {
      this.sp.stage = 'downloading';
      this.sp.target = 55;
      this._setModeBadge('whisper');
    }
    else if (m.includes('загрузка') || m.includes('upload')) {
      this.sp.stage = 'preparing';
      this.sp.target = 40;
    }
    // ── Обнаружение субтитров ────────────────────────────────────
    else if (m.includes('поиск субтитров') || m.includes('проверка наличия субтитров') || (m.includes('检测') && (m.includes('字幕') || m.includes('subtitle')))) {
      this.sp.stage = 'subtitle';
      this.sp.target = 40;
    }
    // ── Другие этапы ─────────────────────────────────────────────
    else if (m.includes('анализ') || m.includes('pars') || m.includes('解析'))          { this.sp.stage = 'parsing';       this.sp.target = 60; }
    else if (m.includes('загрузка') || m.includes('download'))                     { this.sp.stage = 'downloading';   this.sp.target = 60; }
    else if (m.includes('транскрипция') || m.includes('transcrib') || m.includes('whisper') || m.includes('转录')) { this.sp.stage = 'transcribing';  this.sp.target = 80; }
    else if (m.includes('оптимизация') || m.includes('optimiz') || m.includes('优化')) { this.sp.stage = 'optimizing';    this.sp.target = 90; }
    else if (m.includes('резюме') || m.includes('summary') || m.includes('摘要')) { this.sp.stage = 'summarizing';   this.sp.target = 95; }
    else if (m.includes('завершена') || m.includes('complet') || m.includes('完成')) { this.sp.stage = 'completed';     this.sp.target = 100; }

    if (pct >= this.sp.target) this.sp.target = Math.min(pct + 8, 99);
  }

  _setModeBadge(mode) {
    if (!this.modeBadge) return;
    if (mode === 'subtitle') {
      this.modeBadge.textContent  = this.t('mode_subtitle');
      this.modeBadge.className    = 'mode-badge subtitle';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.add('subtitle-mode');
    } else if (mode === 'whisper') {
      this.modeBadge.textContent  = this.t('mode_whisper');
      this.modeBadge.className    = 'mode-badge whisper';
      this.modeBadge.style.display = 'inline-block';
      if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
    }
  }

  _initSP() {
    this.sp.enabled = false; this.sp.current = 0; this.sp.target = 15;
    this.sp.lastServer = 0;  this.sp.startTime = Date.now(); this.sp.stage = 'preparing';
  }
  _startSP() {
    if (this.sp.interval) clearInterval(this.sp.interval);
    this.sp.enabled   = true;
    this.sp.startTime = this.sp.startTime || Date.now();
    this.sp.interval  = setInterval(() => this._tickSP(), 500);
  }
  _stopSP() {
    if (this.sp.interval) { clearInterval(this.sp.interval); this.sp.interval = null; }
    this.sp.enabled = false;
  }
  _tickSP() {
    if (!this.sp.enabled || this.sp.current >= this.sp.target) return;
    const speeds = { subtitle: .5, parsing: .3, downloading: .18, transcribing: .14, optimizing: .22, summarizing: .28 };
    let inc = speeds[this.sp.stage] || .2;
    const remaining = this.sp.target - this.sp.current;
    if (remaining < 5) inc *= .3;
    const next = Math.min(this.sp.current + inc, this.sp.target);
    if (next > this.sp.current) {
      this.sp.current = next;
      this._renderProgress(next, this._stageMsg());
    }
  }
  _stageMsg() {
    const map = {
      subtitle:       this.t('detecting_subtitles'),
      subtitle_found: this.t('subtitle_found'),
      downloading:    this.t('downloading_video'),
      parsing:        this.t('parsing_video'),
      transcribing:   this.t('transcribing_audio'),
      optimizing:     this.t('optimizing_transcript'),
      summarizing:    this.t('generating_summary'),
      completed:      this.t('completed'),
    };
    return map[this.sp.stage] || this.t('processing');
  }

  _renderProgress(pct, msg) {
    const p = Math.round(pct * 10) / 10;
    this.progressStatus.textContent = `${p}%`;
    this.progressFill.style.width   = `${p}%`;

    const m = (msg || '').toLowerCase();
    let label = msg;
    if      (m.includes('субтитры получены') || m.includes('subtitle found') || m.includes('获取成功'))        label = this.t('subtitle_found');
    else if (m.includes('субтитры не найдены') || m.includes('no subtitle') || m.includes('未找到字幕'))         label = this.t('no_subtitle');
    else if (m.includes('поиск субтитров') || (m.includes('检测') && (m.includes('字幕') || m.includes('subtitle')))) label = this.t('detecting_subtitles');
    else if (m.includes('загрузка') || m.includes('download'))  label = this.t('downloading_video');
    else if (m.includes('анализ') || m.includes('pars') || m.includes('解析'))      label = this.t('parsing_video');
    else if (m.includes('транскрипция') || m.includes('transcrib') || m.includes('转录')) label = this.t('transcribing_audio');
    else if (m.includes('оптимизация') || m.includes('optimiz') || m.includes('优化'))   label = this.t('optimizing_transcript');
    else if (m.includes('резюме') || m.includes('summary') || m.includes('摘要'))   label = this.t('generating_summary');
    else if (m.includes('завершена') || m.includes('complet') || m.includes('完成'))   label = this.t('completed');
    else if (m.includes('подготовка') || m.includes('prepar') || m.includes('准备'))    label = this.t('preparing');

    this.progressMessage.textContent = label;
  }

  _showProgress() {
    this.emptyState.style.display    = 'none';
    this.resultsPanel.classList.remove('show');
    this.progressPanel.classList.add('show');
    if (this.modeBadge) { this.modeBadge.style.display = 'none'; this.modeBadge.className = 'mode-badge'; }
    if (this.progressFill) this.progressFill.classList.remove('subtitle-mode');
  }
  _hideProgress() { this.progressPanel.classList.remove('show'); }

  /* ── Результаты ──────────────────────────────────────────── */
  _normLangTab(code) {
    if (!code) return '';
    const c = String(code).toLowerCase().trim();
    if (c.startsWith('zh')) return 'zh';
    if (c.startsWith('ru')) return 'ru';
    if (c.length >= 2) return c.slice(0, 2);
    return c;
  }

  _showResults(script, summary, videoTitle, translation, detectedLang, summaryLang) {
    this.scriptContent.innerHTML  = script    ? marked.parse(script)      : '';
    this.summaryContent.innerHTML = summary   ? marked.parse(summary)     : '';

    const d = this._normLangTab(detectedLang);
    const s = this._normLangTab(summaryLang);
    const showTranslation = Boolean(translation) && d && s && d !== s;
    if (showTranslation) {
      this.translationContent.innerHTML = marked.parse(translation);
      this.translationTabBtn.style.display  = 'inline-block';
      this.dlTranslation.style.display      = 'inline-flex';
    } else {
      this.translationTabBtn.style.display  = 'none';
      this.dlTranslation.style.display      = 'none';
    }

    this.resultsPanel.classList.add('show');
    this._switchTab('script');
    this.resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  _hideResults() { this.resultsPanel.classList.remove('show'); }

  /* ── Вкладки ─────────────────────────────────────────────── */
  _switchTab(name) {
    this.tabBtns.forEach(b  => b.classList.toggle('active',  b.dataset.tab === name));
    this.tabPanes.forEach(p => p.classList.toggle('active', p.id === `${name}Tab`));
  }

  /* ── Загрузка ─────────────────────────────────────────── */
  async _downloadFile(type) {
    if (!this.currentTaskId) { this._showError(this.t('error_no_download')); return; }
    try {
      const r = await fetch(`${this.apiBase}/task-status/${this.currentTaskId}`);
      if (!r.ok) throw new Error('Не удалось получить статус задачи');
      const task = await r.json();

      let filename;
      if      (type === 'script')      filename = task.script_path      ? task.script_path.split('/').pop()      : `transcript_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else if (type === 'summary')     filename = task.summary_path     ? task.summary_path.split('/').pop()     : `summary_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else if (type === 'translation') filename = task.translation_path ? task.translation_path.split('/').pop() : `translation_${task.safe_title||'x'}_${task.short_id||'x'}.md`;
      else throw new Error('Неизвестный тип');

      const a = document.createElement('a');
      a.href = `${this.apiBase}/download/${encodeURIComponent(filename)}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      this._showError(this.t('error_download_failed') + e.message);
    }
  }

  /* ── Вспомогательные функции UI ───────────────────────── */
  _setLoading(on) {
    this.submitBtn.disabled = on;
    this.submitBtn.innerHTML = on
      ? `<span class="spinner"></span> ${this.t('processing')}`
      : `<i class="fas fa-search"></i> <span>${this.t('start_transcription')}</span>`;
    if (this.uploadPickBtn) this.uploadPickBtn.disabled = on;
    if (this.uploadZone) {
      this.uploadZone.style.pointerEvents = on ? 'none' : '';
      this.uploadZone.style.opacity = on ? '0.65' : '';
      this.uploadZone.tabIndex = on ? -1 : 0;
    }
    if (this.fileInput) this.fileInput.disabled = on;
  }

  _showError(msg) {
    this.errorMsg.textContent = msg;
    this.errorBanner.classList.add('show');
    this.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setTimeout(() => this._hideError(), 6000);
  }
  _hideError() { this.errorBanner.classList.remove('show'); }

  _debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }
}

/* ── Запуск ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  window.vt = new VideoTranscriber();
});

window.addEventListener('beforeunload', () => {
  if (window.vt?.eventSource) window.vt._stopSSE();
});