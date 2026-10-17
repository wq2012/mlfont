"""Comprehensive unit tests for the MLFont open-source package."""

from __future__ import annotations

from pathlib import Path
import random

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
import numpy as np

from mlfont import (
    DecayConfig,
    DnnTmConfig,
    DnnTmLearner,
    DocumentCorpusProcessor,
    EvaluationParams,
    ExponentialDecay,
    FontLoadingEvaluator,
    OnlineLdaConfig,
    OnlineLdaLearner,
    PowerDecay,
    RangeCompactConfig,
    RangeCompactGeneticLearner,
    SlicingStrategy,
    balance_topics_diff_constrained,
    balance_topics_greedy,
    decrease_max_topic_mutation,
    export_pretrained_bundle,
    extract_codepoints_from_font,
    fill_codepoint_gaps,
    format_unicode_ranges,
    generate_equal_bin_strategy,
    generate_language_slice_strategy,
    generate_min_slice_strategy,
    generate_most_and_equal_bin_strategy,
    generate_random_slice_strategy,
    generate_single_slice_strategy,
    get_equally_sized_chunks,
    get_preset_config,
    get_slices_based_on_frequency,
    increase_min_topic_mutation,
    init_gene_by_equal_binning,
    init_gene_from_strategy,
    init_gene_from_topic_matrix,
    left_shift_mutation,
    load_pretrained,
    merge_partial_strategies,
    merge_short_range_mutation,
    normalize_lambda_matrix,
    parse_codepoints_file,
    preprocess_codepoint_list_and_lambda_matrix,
    purely_random_crossover,
    purely_random_mutation,
    right_shift_mutation,
    save_codepoints_file,
    save_evaluation_results_json,
    slice_font_file,
    swap_topic_mutation,
)


def _create_tiny_ttf(path: Path, codepoints: list[int]) -> Path:
    """Builds a minimal valid TrueType font supporting the given codepoints."""
    fb = FontBuilder(1000, isTTF=True)
    glyph_order = [".notdef"] + [f"uni{cp:04X}" for cp in codepoints]
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({cp: f"uni{cp:04X}" for cp in codepoints})

    pen = TTGlyphPen(None)
    pen.moveTo((100, 100))
    pen.lineTo((100, 700))
    pen.lineTo((600, 700))
    pen.lineTo((600, 100))
    pen.closePath()
    glyph = pen.glyph()

    glyphs = {gname: glyph for gname in glyph_order}
    fb.setupGlyf(glyphs)
    metrics = {gname: (600, 100) for gname in glyph_order}
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TinyTest", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.save(str(path))
    return path


def test_slicing_strategy_dna_and_serialization(tmp_path: Path) -> None:
    strat = SlicingStrategy(
        subsets=[[0x41, 0x42, 0x43], [0x44, 0x46], [0x45]],
        name="test_strat",
    )
    assert strat.num_slices == 3
    assert strat.num_codepoints == 6
    assert strat.count_unicode_ranges() == 4
    assert strat.count_dna_compactness_ranges() == 4

    dna = strat.to_dna()
    reconstructed = SlicingStrategy.from_dna(dna, num_topics=3, name="reconstructed")
    assert reconstructed.subsets == strat.subsets

    json_path = tmp_path / "strat.json"
    strat.save_json(json_path)
    loaded = SlicingStrategy.load_json(json_path)
    assert loaded.subsets == strat.subsets
    assert loaded.name == "test_strat"

    nam_files = strat.save_nam_dir(tmp_path / "nams")
    assert len(nam_files) == 3
    assert sorted(parse_codepoints_file(nam_files[0])) == [0x41, 0x42, 0x43]


def test_data_parsing_and_corpus_processor(tmp_path: Path) -> None:
    nam_file = tmp_path / "sample.nam"
    nam_file.write_text(
        "# Comment line\n0x0041 A\n0x0042 B\nU+0044 D\n0x0045 E\n",
        encoding="utf-8",
    )
    cps = parse_codepoints_file(nam_file)
    assert cps == {0x41, 0x42, 0x44, 0x45}
    filled = fill_codepoint_gaps(cps, fill_gap=2)
    assert filled == {0x41, 0x42, 0x43, 0x44, 0x45}

    out_nam = tmp_path / "saved.nam"
    save_codepoints_file(filled, out_nam)
    assert parse_codepoints_file(out_nam) == filled

    docs = ["AB", "ABC", "AD", "BCD"]
    proc = DocumentCorpusProcessor(docs, random_seed=0)
    valid_cps = {0x41, 0x42, 0x43, 0x44}
    freqs = proc.count_codepoint_frequency(valid_cps, deduplicate_within_doc=True)
    assert freqs[0x41] == 3  # in "AB", "ABC", "AD"
    assert freqs[0x42] == 3  # in "AB", "ABC", "BCD"
    assert freqs[0x43] == 2  # in "ABC", "BCD"
    assert freqs[0x44] == 2  # in "AD", "BCD"

    batch = proc.construct_batch_sparse_matrix([0x41, 0x42, 0x43, 0x44], n_samples=3)
    assert batch.shape == (3, 4)


def test_normalize_and_topic_balancing() -> None:
    mat = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float64)
    col_norm = normalize_lambda_matrix(mat, "topic_sum")
    np.testing.assert_allclose(np.sum(col_norm, axis=0), np.ones(4))

    row_norm = normalize_lambda_matrix(mat, "row_sum")
    np.testing.assert_allclose(np.sum(row_norm, axis=1), np.ones(2))

    max_norm = normalize_lambda_matrix(mat, "global_max")
    assert np.isclose(np.max(max_norm), 1.0)

    spec_norm = normalize_lambda_matrix(mat, "spectral_norm")
    assert np.isclose(np.linalg.norm(spec_norm, ord=2), 1.0)

    sink_norm = normalize_lambda_matrix(mat, "sinkhorn")
    assert sink_norm.shape == (2, 4)
    np.testing.assert_allclose(np.sum(sink_norm, axis=0), np.ones(4), atol=1e-2)
    np.testing.assert_allclose(np.sum(sink_norm, axis=1), np.array([2.0, 2.0]), atol=1e-2)

    greedy = balance_topics_greedy(mat, "topic_sum")
    assert greedy.shape == (2, 4)
    np.testing.assert_array_equal(np.sum(greedy, axis=0), np.ones(4, dtype=np.uint8))
    np.testing.assert_array_equal(np.sum(greedy, axis=1), np.array([2, 2]))

    diff_c = balance_topics_diff_constrained(mat, max_diff=1, normalization_method="topic_sum")
    assert diff_c.shape == (2, 4)
    np.testing.assert_array_equal(np.sum(diff_c, axis=0), np.ones(4, dtype=np.uint8))


def test_parameter_decay() -> None:
    cfg = DecayConfig(min_val=10.0, max_val=50.0, decay_method="POWER", decay_power=3.0)
    p_decay = PowerDecay(cfg, max_step=100)
    assert np.isclose(p_decay(0), 50.0)
    assert np.isclose(p_decay(100), 10.0)
    assert 10.0 < p_decay(50) < 50.0

    e_cfg = DecayConfig(min_val=10.0, max_val=50.0, decay_method="EXPONENTIAL", decay_rate=0.1)
    e_decay = ExponentialDecay(e_cfg, max_step=100)
    assert np.isclose(e_decay(0), 50.0)
    assert np.isclose(e_decay(100), 10.0)


def test_frequency_slicing() -> None:
    chunks = get_equally_sized_chunks([1, 2, 3, 4, 5], 2)
    assert chunks == [[1, 2, 3], [4, 5]]

    cps = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    freq_arr = np.array([10, 8, 5, 3, 1], dtype=np.int64)

    most_idx, most_s, rem_cps, least_s = get_slices_based_on_frequency(
        cps, most_sizes=(1, 1), frequency_array=freq_arr, least_sizes=(1,)
    )
    assert most_s == [[ord("A")], [ord("B")]]
    assert least_s == [[ord("E")]]
    assert rem_cps == [ord("C"), ord("D")]
    assert len(most_idx) == 2

    mat = np.ones((2, 5), dtype=np.float64)
    rem_cps2, rem_mat = preprocess_codepoint_list_and_lambda_matrix(
        most_sizes=(2,), frequency_array=freq_arr, codepoint_list=cps, lambda_matrix=mat
    )
    assert len(rem_cps2) == 3
    assert rem_mat.shape == (2, 3)

    strat = generate_most_and_equal_bin_strategy(
        cps, frequency_array=freq_arr, most_sizes=(2,), num_bins=2
    )
    assert strat.num_slices == 3
    assert strat.all_codepoints() == set(cps)


def test_online_lda_and_dnntm() -> None:
    cps = list(range(100, 112))
    docs = []
    for _ in range(30):
        docs.append("".join(chr(c) for c in cps[:6]))
        docs.append("".join(chr(c) for c in cps[6:]))
    proc = DocumentCorpusProcessor(docs)

    olda = OnlineLdaLearner(
        proc,
        cps,
        OnlineLdaConfig(n_topics=2, batch_size=16, n_batches=10, random_state=42),
    )
    olda_strat = olda.fit()
    assert olda_strat.num_slices == 2
    assert olda_strat.all_codepoints() == set(cps)
    assert olda.lambda_matrix_ is not None
    assert olda.lambda_matrix_.shape == (2, 12)

    for loss_type in (
        "REAL_FAKE_DIFF",
        "REAL_FAKE_AND_REAL_AVERAGE_DIFF",
        "ALL_COMBINED",
        "AUTOENCODER",
        "HYBRID_AUTOENCODER",
    ):
        dnntm = DnnTmLearner(
            proc,
            cps,
            DnnTmConfig(
                n_topics=2,
                n_layers=2,
                layer_size=16,
                batch_size=16,
                training_steps=15,
                loss_type=loss_type,
                random_seed=42,
            ),
        )
        dnn_strat = dnntm.fit()
        assert dnn_strat.num_slices == 2
        assert dnn_strat.all_codepoints() == set(cps)
        assert dnntm.lambda_matrix_ is not None
        assert dnntm.lambda_matrix_.shape == (2, 12)


def test_range_compact_ga_and_all_mutations() -> None:
    cps = list(range(1000, 1040))
    np_rng = np.random.default_rng(42)
    py_rng = random.Random(42)
    raw_weights = np_rng.uniform(0.1, 1.0, size=(4, 40))
    norm_weights = normalize_lambda_matrix(raw_weights, "topic_sum")

    g_mat = init_gene_from_topic_matrix(cps, norm_weights)
    g_bin = init_gene_by_equal_binning(cps, 4)
    assert g_bin.compute_costs(norm_weights)[0] == 4.0
    assert g_bin.check_no_topic_vanished()

    strat = g_bin.convert_to_strategy()
    g_re = init_gene_from_strategy(strat)
    assert g_re.compute_costs(norm_weights)[0] == 4.0

    for mut_fn in (
        purely_random_mutation,
        left_shift_mutation,
        right_shift_mutation,
        swap_topic_mutation,
        increase_min_topic_mutation,
        decrease_max_topic_mutation,
        merge_short_range_mutation,
    ):
        mutated = mut_fn(g_mat, rng=py_rng)
        assert mutated.n_codepoints == 40

    child = purely_random_crossover(g_mat, g_bin, rng=py_rng)
    assert child.n_codepoints == 40

    ga_cfg = RangeCompactConfig(
        range_cost_weight=0.02,
        max_iteration=60,
        n_survivals=15,
        n_mutations_per_iteration=30,
        n_crossovers_per_iteration=15,
        decay_max_diff=DecayConfig(
            min_val=4.0, max_val=10.0, decay_method="POWER", decay_power=2.0
        ),
        history_interval=20,
        random_seed=42,
    )
    learner = RangeCompactGeneticLearner(
        cps, raw_weights, ga_cfg, normalization_method="topic_sum"
    )
    ga_strat = learner.run()
    assert ga_strat.num_slices == 4
    assert ga_strat.all_codepoints() == set(cps)
    assert len(learner.history_) >= 3
    assert learner.history_[-1].best_final_cost <= learner.history_[0].best_final_cost + 1e-9

    from mlfont.range_compact import build_island_bridged_topics, potts_viterbi_smooth_topics

    t_potts = potts_viterbi_smooth_topics(cps, norm_weights, range_cost_weight=0.05)
    assert len(t_potts) == len(cps)
    assert set(t_potts.tolist()).issubset({0, 1, 2, 3})

    sparse_cps = list(range(0x4E00, 0x4E00 + 5200))
    sparse_mat = np_rng.uniform(0.0, 0.01, size=(10, 5200))
    rem_freq = np.zeros(5200, dtype=np.float64)
    for idx in range(0, 600, 2):
        rem_freq[idx] = 10.0
        sparse_mat[idx % 4, idx] = 1.0
    t_isl = build_island_bridged_topics(sparse_cps, sparse_mat, rem_frequency_array=rem_freq)
    assert t_isl is not None
    assert len(t_isl) == 5200
    assert np.all((t_isl >= 0) & (t_isl < 10))


def test_baselines_subsetter_and_evaluator(tmp_path: Path) -> None:
    cps = list(range(0x41, 0x5B))  # A..Z (26 codepoints)
    s_single = generate_single_slice_strategy(cps)
    s_min = generate_min_slice_strategy(cps)
    s_eq = generate_equal_bin_strategy(cps, num_slices=4)
    s_rand = generate_random_slice_strategy(cps, num_slices=4, seed=7)
    assert s_rand.all_codepoints() == set(cps)

    nam1 = tmp_path / "vowels.nam"
    save_codepoints_file([ord(c) for c in "AEIOU"], nam1)
    s_lang = generate_language_slice_strategy(cps, [nam1], include_unassigned=True)
    assert s_lang.num_slices == 2
    assert s_lang.all_codepoints() == set(cps)

    merged = merge_partial_strategies(
        [generate_single_slice_strategy(cps[:5], "part1")], full_codepoints=cps
    )
    assert merged.num_slices == 2
    assert merged.all_codepoints() == set(cps)

    uranges = format_unicode_ranges([0x41, 0x42, 0x43, 0x45, 0x47, 0x48])
    assert uranges == ["U+41-43", "U+45", "U+47-48"]

    ttf_path = _create_tiny_ttf(tmp_path / "TinyTest.ttf", cps)
    extracted_cps = extract_codepoints_from_font(ttf_path)
    assert extracted_cps == cps

    artifacts, css_path = slice_font_file(
        ttf_path, s_eq, tmp_path / "sliced", flavor="woff2"
    )
    assert len(artifacts) == 4
    assert all(a.byte_size > 0 and a.path.exists() for a in artifacts)
    assert "@font-face" in css_path.read_text(encoding="utf-8")

    samples = [
        {ord("A"), ord("B"), ord("C")},
        {ord("A"), ord("D"), ord("E")},
        {ord("F"), ord("G")},
        {ord("X"), ord("Y"), ord("Z")},
    ]
    evaluator = FontLoadingEvaluator(cps, samples)
    params = EvaluationParams(
        name="test_3g",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,
        average_glyph_size_bytes=200.0,
        session_length=2,
    )
    res_single = evaluator.evaluate_strategy(s_single, params)
    res_min = evaluator.evaluate_strategy(s_min, params)
    res_eq = evaluator.evaluate_strategy(s_eq, params)

    assert res_single.num_missing_codepoints == 0
    assert res_min.upfront_utilization_pct == 100.0
    assert res_min.session_utilization_pct == 100.0
    assert res_eq.num_ranges == 4

    out_eval_json = tmp_path / "eval_results.json"
    save_evaluation_results_json([res_single, res_min, res_eq], out_eval_json)
    assert out_eval_json.exists()

    for preset_name in ("chinese", "traditional_chinese", "japanese", "korean", "lgc"):
        preset = get_preset_config(preset_name)
        assert preset.olda_config.n_topics > 0

    bundle_dir = export_pretrained_bundle(
        output_dir=tmp_path / "pretrained_bundle",
        strategy=s_eq,
        lambda_matrix=np.ones((4, 26), dtype=np.float64),
        codepoints=cps,
        frequency_array=np.arange(26, dtype=np.int64),
        font_path=ttf_path,
        font_family="TinyTest Sliced",
        slice_font=True,
        evaluation_results=[res_eq],
    )
    loaded_bundle = load_pretrained(bundle_dir)
    assert loaded_bundle.strategy.num_slices == 4
    assert loaded_bundle.lambda_matrix is not None
    assert loaded_bundle.lambda_matrix.shape == (4, 26)
    assert loaded_bundle.predict_required_slices("AZ") == [0, 3]
    assert (bundle_dir / "font_sliced_cdn.css").exists()
    assert "font-display: swap;" in (bundle_dir / "font_sliced_cdn.css").read_text(encoding="utf-8")
    assert (bundle_dir / "mlfont.js").exists()
    readme_text = (bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "Using This Sliced Font in Your Website" in readme_text
    assert "mlfont.js" in readme_text


def test_sparse_vocabulary_max_diff_and_aliases(tmp_path: Path) -> None:
    import mlfont

    for sym in (
        "DNNTMConfig",
        "DNNTMSolver",
        "FontSubsetter",
        "FrequencySolver",
        "OnlineLDAConfig",
        "OnlineLDASolver",
        "RangeCompactGASolver",
        "extract_font_codepoints",
        "normalize_topic_matrix",
        "save_safetensors_model",
    ):
        assert hasattr(mlfont, sym), f"Missing public symbol {sym}"

    # 120 codepoints where only 40 are observed in training documents
    cps = list(range(0x4E00, 0x4E00 + 120))
    docs = []
    for i in range(40):
        start_c = (i % 4) * 10
        end_c = ((i % 4) + 1) * 10
        cluster = cps[start_c:end_c]
        docs.append("".join(chr(c) for c in cluster))
    proc = DocumentCorpusProcessor(docs, random_seed=42)

    olda = OnlineLdaLearner(
        proc,
        cps,
        OnlineLdaConfig(
            n_topics=6,
            most_sizes=[12],
            normalization_method="global_max",
            batch_size=20,
            n_batches=5,
            random_state=42,
        ),
    )
    s_olda = olda.fit()
    assert s_olda.num_slices == 7  # 1 most_slice + 6 topic slices
    assert olda.lambda_matrix_ is not None
    assert olda.lambda_matrix_.shape == (6, 108)
    # All 6 topic rows carry real mass (> 1e-3) and non-constant learned signal on active cols
    assert int(np.sum(np.any(olda.lambda_matrix_ > 1e-3, axis=1))) == 6
    assert int(np.sum(np.ptp(olda.lambda_matrix_[:, :28], axis=1) > 1e-4)) == 6
    assert int(np.sum(np.max(olda.lambda_matrix_, axis=0) <= 1e-3)) == 0

    dnntm = DnnTmLearner(
        proc,
        cps,
        DnnTmConfig(
            n_topics=6,
            most_sizes=[12],
            normalization_method="global_max",
            batch_size=16,
            n_layers=2,
            layer_size=16,
            training_steps=20,
            loss_type="ALL_COMBINED",
            random_seed=42,
        ),
    )
    s_dnn = dnntm.fit()
    assert s_dnn.num_slices == 7
    assert all(loss_val >= 0.0 for loss_val in dnntm.loss_history_)

    freq = proc.count_frequency_array(cps)
    ga_cfg = RangeCompactConfig(
        range_cost_weight=0.005,
        max_iteration=80,
        n_survivals=15,
        n_mutations_per_iteration=30,
        n_crossovers_per_iteration=15,
        decay_max_diff=DecayConfig(
            min_val=3.0, max_val=15.0, decay_method="POWER", decay_power=3.0
        ),
        random_seed=42,
    )
    ga = RangeCompactGeneticLearner(
        cps,
        olda.lambda_matrix_,
        ga_cfg,
        frequency_array=freq,
        most_sizes=[12],
        normalization_method="global_max",
    )
    s_ga = ga.run()
    topic_slice_sizes = [len(s) for s in s_ga.subsets[1:]]
    assert max(topic_slice_sizes) - min(topic_slice_sizes) <= 3


def test_readme_quickstart_api_and_edge_cases(tmp_path: Path) -> None:
    from mlfont import (
        DNNTMConfig,
        DNNTMSolver,
        FontSubsetter,
        FrequencySolver,
        OnlineLDAConfig,
        OnlineLDASolver,
        RangeCompactConfig,
        RangeCompactGASolver,
        extract_font_codepoints,
    )

    cps = list(range(0x0041, 0x0041 + 60))
    font_path = _create_tiny_ttf(tmp_path / "Quickstart.ttf", cps)
    font_codepoints = extract_font_codepoints(font_path)
    assert len(font_codepoints) == 60

    train_docs = [
        set(font_codepoints[:20]),
        set(font_codepoints[15:35]),
        set(font_codepoints[30:50]),
    ] * 10

    # README quickstart form (explicit keyword); the positional form is kept as a
    # backward-compatible heuristic for long codepoint lists.
    freq_solver = FrequencySolver(codepoints=font_codepoints)
    assert FrequencySolver(font_codepoints).codepoints_ == freq_solver.codepoints_
    freq_solver.fit(train_docs)
    most_slices, remaining_codepoints = freq_solver.partition_most_frequent(
        most_counts=[12]
    )
    assert len(most_slices) == 1
    assert len(most_slices[0]) == 12
    assert len(remaining_codepoints) == 48

    olda_solver = OnlineLDASolver(
        codepoints=remaining_codepoints,
        config=OnlineLDAConfig(
            n_topics=4,
            batch_size=16,
            max_iter=5,
            normalization="global_max",
            random_state=42,
        ),
    )
    topic_matrix = olda_solver.fit_transform(train_docs)
    assert topic_matrix.shape == (4, 48)

    dnntm_solver = DNNTMSolver(
        codepoints=remaining_codepoints,
        config=DNNTMConfig(
            n_topics=4,
            batch_size=16,
            max_iter=10,
            normalization="global_max",
            random_state=42,
        ),
    )
    dnn_matrix = dnntm_solver.fit_transform(train_docs)
    assert dnn_matrix.shape == (4, 48)

    # Test exact max_diff=0 when 48 % 4 == 0
    ga_solver = RangeCompactGASolver(
        codepoints=remaining_codepoints,
        topic_matrix=topic_matrix,
        config=RangeCompactConfig(
            population_size=10,
            num_iterations=40,
            decay_power=3.0,
            range_cost_weight=0.012,
            decay_max_diff=DecayConfig(min_val=0.0, max_val=4.0),
            random_state=42,
        ),
    )
    strategy = ga_solver.optimize(
        strategy_name="range_compact_most12_topics4",
        font_name="Quickstart",
        prefix_slices=most_slices,
    )
    assert strategy.num_slices == 5
    assert strategy.total_num_ranges() == strategy.num_ranges
    ga_sizes = [len(s) for s in strategy.subsets[1:]]
    assert max(ga_sizes) - min(ga_sizes) == 0

    subsetter = FontSubsetter(font_path)
    artifacts, css_path = subsetter.slice_font(strategy, tmp_path / "out_woff2")
    assert len(artifacts) == 5
    assert css_path.exists()

    # Edge case: V_active < K (only 2 active codepoints, K=6 topics)
    tiny_docs = [
        "".join(chr(c) for c in remaining_codepoints[:2]),
        "".join(chr(c) for c in remaining_codepoints[:2]),
    ]
    sparse_solver = OnlineLDASolver(
        codepoints=remaining_codepoints,
        config=OnlineLDAConfig(n_topics=6, batch_size=4, max_iter=3, random_state=42),
    )
    sparse_strat = sparse_solver.fit(tiny_docs)
    assert sparse_strat.num_slices == 6
    assert sparse_strat.all_codepoints() == set(remaining_codepoints)


def test_cli_scripts_end_to_end(tmp_path: Path, monkeypatch) -> None:
    import sys
    from scripts import evaluate, export_model, inference, prepare_data, train

    font_path = _create_tiny_ttf(tmp_path / "CliFont.ttf", list(range(0x0041, 0x0041 + 40)))
    data_dir = tmp_path / "data"
    train_dir = tmp_path / "trained"
    export_dir = tmp_path / "exported"
    eval_json = tmp_path / "eval_results.json"
    slice_dir = tmp_path / "sliced"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-prepare-data",
            "--font_path",
            str(font_path),
            "--generate_synthetic",
            "--num_synthetic",
            "40",
            "--output_dir",
            str(data_dir),
        ],
    )
    prepare_data.main()
    assert (data_dir / "codepoints.nam").exists()
    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "test.jsonl").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-train",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(train_dir),
            "--n_topics",
            "4",
            "--most_sizes",
            "8",
            "--max_iteration",
            "15",
        ],
    )
    train.main()
    assert (train_dir / "strategy.json").exists()
    assert (train_dir / "model.safetensors").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-evaluate",
            "--codepoints_file",
            str(data_dir / "codepoints.nam"),
            "--test_jsonl",
            str(data_dir / "test.jsonl"),
            "--strategy_files",
            str(train_dir / "strategy.json"),
            "--output_json",
            str(eval_json),
        ],
    )
    evaluate.main()
    assert eval_json.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-export",
            "--checkpoint_dir",
            str(train_dir),
            "--export_dir",
            str(export_dir),
            "--font_path",
            str(font_path),
        ],
    )
    export_model.main()
    assert (export_dir / "model.safetensors").exists()
    assert (export_dir / "strategy.json").exists()
    assert (export_dir / "font_sliced.css").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-slice",
            "--model_dir",
            str(export_dir),
            "--font_path",
            str(font_path),
            "--output_dir",
            str(slice_dir),
            "--text",
            "ABCDEF",
        ],
    )
    inference.main()
    assert (slice_dir / "clifont_sliced.css").exists()

    # Also test mlfont-train with --normalization_method sinkhorn and --loss_type
    train_dnn_dir = tmp_path / "trained_dnn"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-train",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(train_dnn_dir),
            "--algorithm",
            "dnntm",
            "--n_topics",
            "4",
            "--most_sizes",
            "8",
            "--normalization_method",
            "sinkhorn",
            "--loss_type",
            "HYBRID_AUTOENCODER",
        ],
    )
    train.main()
    assert (train_dnn_dir / "strategy.json").exists()


def test_public_exports_and_skewed_edge_cases(tmp_path: Path) -> None:
    import random
    import mlfont
    from mlfont.range_compact import (
        build_island_bridged_topics,
        potts_viterbi_smooth_topics,
    )

    assert hasattr(mlfont, "DNNTMLossType")
    assert hasattr(mlfont, "DnnTmLossType")
    assert hasattr(mlfont, "sinkhorn_knopp_normalize")
    assert hasattr(mlfont, "slice_font_to_woff2")

    mat = np.array([[2.0, 1.0, 3.0], [1.0, 4.0, 1.0]], dtype=np.float64)
    sk = mlfont.sinkhorn_knopp_normalize(mat)
    np.testing.assert_allclose(sk.sum(axis=0), np.ones(3), atol=1e-3)
    np.testing.assert_allclose(sk.sum(axis=1), np.full(2, 1.5), atol=1e-3)

    # Skewed matrix: K=50, N=10000 where row 0 dominates everywhere
    skewed = np.full((50, 10000), 1e-6, dtype=np.float64)
    skewed[0, :] = 1.0
    cps_10k = list(range(10000))
    smoothed = potts_viterbi_smooth_topics(cps_10k, skewed, range_cost_weight=0.18)
    counts = np.bincount(smoothed, minlength=50)
    assert int((counts == 0).sum()) == 0
    assert int(counts.max() - counts.min()) <= 1

    ga_skewed = RangeCompactGeneticLearner(
        codepoint_list=cps_10k,
        lambda_matrix=skewed,
        normalization_method="none",
    )
    repaired = ga_skewed._repair_max_diff(
        np.zeros(10000, dtype=np.int64), max_diff=0, rng=random.Random(42)
    )
    rep_counts = np.bincount(repaired, minlength=50)
    assert int((rep_counts == 0).sum()) == 0
    assert int(rep_counts.max() - rep_counts.min()) == 0
    num_runs = int(1 + np.sum(repaired[1:] != repaired[:-1]))
    assert num_runs == 50

    # Small sparse island bridging (N=100, K=6) and V_rem = K + 1 aspect ratio
    small_sparse = np.full((6, 100), 1e-4, dtype=np.float64)
    for k in range(6):
        small_sparse[k, k * 5:(k + 1) * 5] = 0.9
    bridged = build_island_bridged_topics(list(range(100)), small_sparse)
    assert bridged is not None
    b_counts = np.bincount(bridged, minlength=6)
    assert int(b_counts.max() - b_counts.min()) <= 1

    cps_7 = list(range(0x0041, 0x0041 + 7))
    lam_6x7 = np.random.default_rng(7).uniform(0.1, 1.0, size=(6, 7))
    ga_tiny = RangeCompactGeneticLearner(
        codepoint_list=cps_7,
        lambda_matrix=lam_6x7,
        config=RangeCompactConfig(
            max_iteration=20,
            decay_max_diff=DecayConfig(min_val=1.0, max_val=2.0),
        ),
    )
    strat_tiny = ga_tiny.run()
    assert strat_tiny.num_slices == 6
    tiny_sizes = [len(s) for s in strat_tiny.subsets]
    assert max(tiny_sizes) - min(tiny_sizes) <= 1


def test_website_simulator_html_extraction_and_profiles(tmp_path):
    """Test HTML text/codepoint extraction, file loading, and network profiles."""
    import pytest
    from mlfont.simulator import (
        BrowserNetworkProfile,
        extract_visible_text_from_html,
        extract_webpage_codepoints,
        get_network_profile,
        get_real_website_benchmark_pages,
        load_webpage_sample,
    )

    p_3g = get_network_profile("gin-3g")
    assert p_3g.name == "GIN-3g"
    assert p_3g.bandwidth_kbps == 400.0
    assert p_3g.rtt_ms == 400.0

    # Verify documented network profile aliases resolve properly (GIN-2g-poor must
    # match schemas.STANDARD_EVALUATION_PARAMS: 20 kbps / 1600 ms RTT)
    assert get_network_profile("GIN-2g-poor").bandwidth_kbps == 20.0
    assert get_network_profile("GIN-2g-poor").rtt_ms == 1600.0
    assert get_network_profile("Broadband-20Mbps").bandwidth_kbps == 20000.0
    assert get_network_profile("residential-ADSL2+").bandwidth_kbps == 20000.0

    custom_p = BrowserNetworkProfile(
        "Custom-WiFi", rtt_ms=15.0, bandwidth_bytes_per_ms=2500.0, max_parallel_requests=6
    )
    assert get_network_profile(custom_p) is custom_p
    assert custom_p.rtt_ms == 15.0
    assert custom_p.bandwidth_kbps == 20000.0

    with pytest.raises(ValueError, match="Unknown network profile"):
        get_network_profile("nonexistent-network-999")

    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>忽略标题</title>
      <style>body { font-family: sans-serif; content: "隐藏样式"; }</style>
      <script>const secret = "隐藏脚本";</script>
    </head>
    <body>
      <!-- HTML 注释内容不应被提取 -->
      <h1>开发者门户 &amp; WebFont 测试</h1>
      <p>第一段：机器学习字体分片。</p>
      <noscript>禁用JS提示</noscript>
      <template><p>模板内容</p></template>
    </body>
    </html>
    """
    vis_text = extract_visible_text_from_html(html)
    assert "开发者门户 & WebFont 测试" in vis_text
    assert "第一段：机器学习字体分片。" in vis_text
    assert "隐藏样式" not in vis_text
    assert "隐藏脚本" not in vis_text
    assert "注释内容" not in vis_text
    assert "禁用JS提示" not in vis_text
    assert "模板内容" not in vis_text

    cps = extract_webpage_codepoints(html)
    assert ord("开") in cps
    assert ord("&") in cps
    assert 0x20 in cps  # Collapsed whitespace includes ASCII space
    assert ord("隐") not in cps

    # Plain text mode, valid_codepoints filter, and empty input edge case
    assert extract_webpage_codepoints("") == set()
    plain_cps = extract_webpage_codepoints("Hello\nWorld")
    assert 0x20 in plain_cps
    assert ord("H") in plain_cps
    assert ord("\n") not in plain_cps
    filtered_cps = extract_webpage_codepoints("Hello\nWorld", valid_codepoints={ord("H")})
    assert filtered_cps == {ord("H")}

    # Edge cases in HTML extraction:
    # 1) Comment-only or doctype-only HTML should extract to empty string (not raw comment)
    assert extract_visible_text_from_html("<!-- hidden comment -->") == ""
    assert extract_webpage_codepoints("<!-- hidden comment -->") == set()
    assert extract_visible_text_from_html("<!DOCTYPE html><!-- only doctype -->") == ""

    # 2) Valid HTML5 without <head> but with <title> should ignore <title>
    no_head_html = (
        "<!DOCTYPE html><html><title>HiddenTabTitle</title>"
        "<body><p>VisibleBody</p></body></html>"
    )
    assert extract_visible_text_from_html(no_head_html) == "VisibleBody"
    assert ord("H") not in extract_webpage_codepoints(no_head_html)

    # 3) Malformed HTML with unclosed <head> and no <body> should still extract visible body tags
    unclosed_head = "<head><meta charset='utf-8'><title>Tab</title><p>RecoveredText</p>"
    assert extract_visible_text_from_html(unclosed_head) == "RecoveredText"

    # 4) Multi-element HTML without spaces should not inject synthetic U+0020 into codepoints
    emoji_multi = "<p>🚀</p><p>🔥</p>"
    assert 0x20 not in extract_webpage_codepoints(emoji_multi)
    cjk_multi = "<p>天地</p><p>玄黄</p>"
    assert 0x20 not in extract_webpage_codepoints(cjk_multi)

    # 5) Long inline HTML/text string (>255 UTF-8 bytes) must not raise OSError(ENAMETOOLONG)
    long_inline_html = "<article>" + ("机器学习网页字体分片优化测试内容" * 40) + "</article>"
    assert len(long_inline_html.encode("utf-8")) > 1000
    long_sample = load_webpage_sample(long_inline_html, page_id="long_inline")
    assert long_sample.url == "inline://html"
    assert ord("机") in long_sample.codepoints

    # 6) Passing a nonexistent Path object must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_webpage_sample(tmp_path / "does_not_exist_12345.html")

    # Load from .html and .txt files
    html_file = tmp_path / "sample_page.html"
    html_file.write_text(html, encoding="utf-8")
    loaded_html = load_webpage_sample(str(html_file), page_id="sample_page")
    assert loaded_html.page_id == "sample_page"
    assert ord("开") in loaded_html.codepoints

    txt_file = tmp_path / "plain_note.txt"
    txt_file.write_text("纯文本测试内容", encoding="utf-8")
    loaded_txt = load_webpage_sample(str(txt_file), title="Plain Note")
    assert loaded_txt.title == "Plain Note"
    assert ord("纯") in loaded_txt.codepoints

    # Benchmark catalog across all 5 fonts
    all_pages = get_real_website_benchmark_pages("all")
    assert len(all_pages) == 20
    sc_pages = get_real_website_benchmark_pages("notosanssc")
    assert len(sc_pages) == 8
    with pytest.raises(ValueError, match="Unknown font_key"):
        get_real_website_benchmark_pages("invalid_font")


def test_website_font_simulator_and_cli(tmp_path):
    """Test WebsiteFontSimulator page load, warm session, comparison, and CLI."""
    import json
    import pytest
    from mlfont import SlicingStrategy
    from mlfont.simulator import (
        BrowserNetworkProfile,
        WebPageSample,
        WebsiteFontSimulator,
        get_real_website_benchmark_pages,
        save_website_simulation_report_json,
    )
    from scripts.simulate_website import main as simulate_cli_main

    # Build a 3-slice strategy vs equal_bin_3 vs 1-slice full strategy
    s0 = set(range(0x0020, 0x0080))
    s1 = set(range(0x4E00, 0x4E50))
    s2 = set(range(0x4E50, 0x4F00))
    all_cps = sorted(s0 | s1 | s2)
    sliced_strat = SlicingStrategy(subsets=[s0, s1, s2], name="mlfont_3slice")
    eq_strat = SlicingStrategy(subsets=[s0, s1, s2], name="equal_bin_3")
    full_strat = SlicingStrategy(subsets=[s0 | s1 | s2], name="single_slice")

    sim = WebsiteFontSimulator(
        font_key="notosanssc",
        codepoints=all_cps,
        strategies=[full_strat, sliced_strat, eq_strat],
        slice_byte_sizes_map={
            "single_slice": [100_000],
            "mlfont_3slice": [10_000, 40_000, 50_000],
            "equal_bin_3": [10_000, 40_000, 50_000],
        },
        full_font_woff2_bytes=100_000,
    )

    assert sim._get_high_freq_slice_indices("single_slice", full_strat) == {0}
    assert sim._get_high_freq_slice_indices("equal_bin_3", eq_strat) == set()
    assert sim._get_high_freq_slice_indices("mlfont_3slice", sliced_strat) == {0, 1}

    # Verify high_freq_slice_count=0 is respected
    sim_no_hf = WebsiteFontSimulator(
        font_key="notosanssc",
        codepoints=all_cps,
        strategies=[sliced_strat],
        high_freq_slice_count=0,
    )
    assert sim_no_hf._get_high_freq_slice_indices("mlfont_3slice", sliced_strat) == set()

    with pytest.raises(ValueError, match="Unknown strategy"):
        sim.simulate_page_load(
            WebPageSample("p0", "P0", "u", "Cat", "notosanssc", "<p>Hi</p>"),
            "nonexistent_strategy",
            "GIN-3g",
        )

    # Page using all 3 slices so equal_bin_3 Wave-1 downloads all 3 slices
    # while mlfont Tier-1 is s0+s1
    page_all3 = WebPageSample(
        page_id="p_all3",
        title="Page All 3",
        url="https://example.com/all3",
        category="Test",
        font_key="notosanssc",
        html_content="<p>Hello 一丁七 乐乒乓</p>",
    )
    res_ml_all3 = sim.simulate_page_load(page_all3, "mlfont_3slice", "GIN-3g")
    res_eq_all3 = sim.simulate_page_load(page_all3, "equal_bin_3", "GIN-3g")
    assert res_ml_all3.high_freq_token_coverage_pct > 0.0
    assert res_eq_all3.high_freq_token_coverage_pct == 0.0
    assert res_ml_all3.first_text_paint_ms < res_eq_all3.first_text_paint_ms

    # Custom profile with max_parallel_requests=1: 2 HF slices require 2 RTT waves
    serial_prof = BrowserNetworkProfile(
        "Serial-1", rtt_ms=100.0, bandwidth_bytes_per_ms=1000.0, max_parallel_requests=1
    )
    res_serial = sim.simulate_page_load(page_all3, "mlfont_3slice", serial_prof)
    # 2 HF slices (10KB + 40KB = 50KB -> 50ms) + 2 RTTs (200ms) + CSS (~100ms) > 350ms
    assert res_serial.first_text_paint_ms >= 350.0

    # Warm load where HF slices {0, 1} are already cached and only tail slice {2} is downloaded:
    # First text paint should be immediate (0.0 ms when css_cached=True)
    res_warm_tail = sim.simulate_page_load(
        page_all3,
        "mlfont_3slice",
        "GIN-3g",
        cached_slices={0, 1},
        css_cached=True,
        session_page_index=1,
    )
    assert res_warm_tail.downloaded_slice_indices == [2]
    assert res_warm_tail.first_text_paint_ms == 0.0
    assert res_warm_tail.total_font_load_ms > 0.0

    # Page using only s0 and s1 (not s2)
    page1 = WebPageSample(
        page_id="p1",
        title="Page 1",
        url="https://example.com/1",
        category="Test",
        font_key="notosanssc",
        html_content="<p>Hello 一丁七</p>",
    )
    res_sliced = sim.simulate_page_load(page1, "mlfont_3slice", "GIN-3g")
    res_full = sim.simulate_page_load(page1, "single_slice", "GIN-3g")

    assert res_sliced.required_slice_indices == [0, 1]
    assert res_sliced.downloaded_slice_indices == [0, 1]
    assert res_sliced.font_bytes_downloaded == 50_000
    assert res_full.font_bytes_downloaded == 100_000
    assert res_sliced.total_font_load_ms < res_full.total_font_load_ms
    assert res_sliced.first_text_paint_ms <= res_sliced.total_font_load_ms

    # Warm cache session: second load of same slices should take 0 ms and 0 font bytes
    page2 = WebPageSample(
        page_id="p2",
        title="Page 2",
        url="https://example.com/2",
        category="Test",
        font_key="notosanssc",
        html_content="<p>World 万丈三</p>",
    )
    session_results = sim.simulate_browsing_session([page1, page2], "mlfont_3slice", "GIN-3g")
    assert len(session_results) == 2
    assert session_results[1].num_slices_cached == 2
    assert session_results[1].downloaded_slice_indices == []
    assert session_results[1].font_bytes_downloaded == 0
    assert session_results[1].total_font_load_ms == 0.0

    # Single-page compare_strategies: warm_total_load_ms should reflect warm revisit (0.0 ms)
    single_page_rep = sim.compare_strategies(pages=[page1], profiles=["GIN-3g"])
    for s in single_page_rep.summaries:
        assert s.warm_total_load_ms.mean == 0.0
        assert s.warm_slices_downloaded_mean == 0.0

    # Empty page codepoints edge case
    empty_page = WebPageSample(
        page_id="empty",
        title="Empty",
        url="https://example.com/empty",
        category="Test",
        font_key="notosanssc",
        html_content="",
    )
    res_empty = sim.simulate_page_load(empty_page, "mlfont_3slice", "GIN-3g")
    assert res_empty.num_slices_downloaded == 0
    assert res_empty.font_bytes_downloaded == 0
    assert res_empty.high_freq_token_coverage_pct == 0.0

    # Compare strategies & save JSON report
    report = sim.compare_strategies(
        pages=[page1, page2],
        profiles=["GIN-3g", "Cable-5Mbps"],
    )
    assert len(report.summaries) == 6
    mlfont_3g = [
        s for s in report.summaries
        if s.strategy_name == "mlfont_3slice" and s.profile_name == "GIN-3g"
    ][0]
    assert mlfont_3g.speedup_vs_single_slice_cold > 1.2
    assert mlfont_3g.bytes_saved_vs_single_slice_cold_pct > 45.0

    json_out = tmp_path / "sim_report.json"
    save_website_simulation_report_json([report], [], str(json_out))
    assert json_out.exists()
    parsed = json.loads(json_out.read_text(encoding="utf-8"))
    assert "fonts" in parsed and "notosanssc" in parsed["fonts"]

    # Test CLI entrypoint on Noto Sans LGC with --profiles alias and mlfont-* font_key normalization
    cli_json = tmp_path / "cli_sim.json"
    lgc_pages = get_real_website_benchmark_pages("notosans_lgc")
    sample_html_path = tmp_path / "custom_lgc.html"
    sample_html_path.write_text(lgc_pages[0].html_content, encoding="utf-8")

    simulate_cli_main([
        "--font_key", "mlfont-notosans-lgc",
        "--profiles", "GIN-2g-poor,Cable-5Mbps",
        "--html_file", str(sample_html_path),
        "--output_json", str(cli_json),
    ])
    assert cli_json.exists()

    # Test CLI with --text "" (empty inline string)
    empty_cli_json = tmp_path / "empty_cli_sim.json"
    simulate_cli_main([
        "--font_key", "notosans_lgc",
        "--profiles", "Cable-5Mbps",
        "--text", "",
        "--output_json", str(empty_cli_json),
    ])
    empty_rep = json.loads(empty_cli_json.read_text(encoding="utf-8"))
    assert empty_rep["fonts"]["notosans_lgc"]["summaries"][0]["num_pages_evaluated"] == 1


def test_mlfont_js_api_and_aliases() -> None:
    """Verify web/mlfont.js and pretrained_models/*/mlfont.js expose documented API."""
    from scripts.run_noto_experiments import FONT_SPECS, REPO_ROOT

    web_js_path = REPO_ROOT / "web" / "mlfont.js"
    web_js = web_js_path.read_text(encoding="utf-8")
    for alias in (
        "'notosanssc'",
        "'sc'",
        "'zh-cn'",
        "'zh-hans'",
        "'notosanstc'",
        "'tc'",
        "'zh-tw'",
        "'zh-hant'",
        "'notosansjp'",
        "'jp'",
        "'ja'",
        "'notosanskr'",
        "'kr'",
        "'ko'",
        "'notosans-lgc'",
        "'notosans_lgc'",
        "'notosans'",
        "'lgc'",
    ):
        assert alias in web_js, f"Missing documented alias {alias} in web/mlfont.js"

    for prop in (
        "totalSlicesFetched",
        "cachedSlicesCount",
        "totalTransferBytes",
        "maxSliceDurationMs",
    ):
        assert prop in web_js, f"Missing documented metrics.summary.{prop} in web/mlfont.js"

    for spec in FONT_SPECS:
        hf_js = (REPO_ROOT / "pretrained_models" / str(spec["hf_name"]) / "mlfont.js").read_text(
            encoding="utf-8"
        )
        assert hf_js == web_js


def test_hf_model_card_website_simulation_section(tmp_path: Path) -> None:
    """Verify write_hf_model_card and all 5 pretrained model cards include subsection 5."""
    from mlfont import SlicingStrategy
    from scripts.run_noto_experiments import (
        FONT_SPECS,
        REPO_ROOT,
        _build_website_sim_section,
        _load_eval_results_json,
        write_hf_model_card,
    )

    # 1. Verify all 5 checked-in model cards contain subsection 5 and no raw $ math
    for spec in FONT_SPECS:
        hf_name = str(spec["hf_name"])
        key = str(spec["key"])
        card_path = REPO_ROOT / "pretrained_models" / hf_name / "README.md"
        assert card_path.exists()
        content = card_path.read_text(encoding="utf-8")
        assert (
            "### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification"
            in content
        )
        assert f"mlfont-simulate --font_key {key} --html_file" in content
        assert "web/demo.html" in content
        assert "Real Headless Chrome" in content
        assert "https://x.com/googlefonts/status/900019727732531200" in content
        assert "https://www.tdcommons.org/dpubs_series/906/" in content
        assert "$" not in content

    # 2. Test write_hf_model_card in a temporary bundle directory
    spec0 = FONT_SPECS[0]
    bdir0 = REPO_ROOT / "pretrained_models" / str(spec0["hf_name"])
    strat0 = SlicingStrategy.load_json(bdir0 / "strategy.json")
    eval_res0 = _load_eval_results_json(bdir0 / "evaluation_metrics.json")
    ga_res0 = [r for r in eval_res0 if r.strategy_name == strat0.name]
    write_hf_model_card(
        bundle_dir=tmp_path,
        spec=spec0,
        strategy=strat0,
        eval_results=ga_res0,
        all_strat_results=eval_res0,
        total_woff2_bytes=444484,
        css_bytes=7974,
        wall_time_sec=24.0,
    )
    gen_card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert (
        "### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification"
        in gen_card
    )
    assert "lgc_scientific_typography" in gen_card

    # 3. Edge case: missing simulation report path still produces CLI/demo instructions cleanly
    fallback_lines = _build_website_sim_section(
        key="notosans_lgc",
        hf_name="mlfont-notosans-lgc",
        title="Noto Sans (Pan-European LGC)",
        strategy_name="range_compact_topics20",
        sim_report_path=tmp_path / "nonexistent_report.json",
    )
    fallback_text = "\n".join(fallback_lines)
    assert (
        "### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification"
        in fallback_text
    )
    assert "mlfont-simulate --font_key notosans_lgc" in fallback_text


def test_k_eff_report_matches_alignment_rule() -> None:
    """Regression: OnlineLdaLearner.k_eff_ / num_core_codepoints_ must use the
    exact same core rule as align_and_smooth_topic_matrix (previously fit() used
    a different min_df rule and under-reported K_eff on small repertoires)."""
    from mlfont.olda import compute_core_mask_and_k_eff

    # 60 tail codepoints, K=3 -> row_cap = 20. 45 active codepoints: 5 with DF=10
    # and 40 singletons (DF=1). The old reporting rule (min_df=2 whenever
    # n_active > row_cap) gave n_core=5 -> K_eff=1, while the alignment code
    # (min_df=1 for n_rem <= 5000) uses n_core=45 -> K_eff=3.
    cps = list(range(0x100, 0x100 + 60))
    docs = []
    for _ in range(10):
        docs.append("".join(chr(c) for c in cps[:5]))
    for j in range(40):
        docs.append(chr(cps[5 + j]) + chr(cps[0]))
    proc = DocumentCorpusProcessor(docs)
    freq = proc.count_frequency_array(cps, deduplicate_within_doc=True)
    active_df = freq[freq > 0]
    core_mask, n_core, k_eff = compute_core_mask_and_k_eff(active_df, 60, 3)
    assert int(np.sum(core_mask)) == n_core == 45
    assert k_eff == 3

    learner = OnlineLdaLearner(
        proc, cps, OnlineLdaConfig(n_topics=3, batch_size=16, n_batches=5, random_state=0)
    )
    learner.fit()
    assert learner.num_active_codepoints_ == 45
    assert learner.num_core_codepoints_ == n_core
    assert learner.k_eff_ == k_eff
    # Tier-B tail rows are bounded by prior(0.0045) + 0.0018 + 0.0035 < 0.0105,
    # so the number of rows carrying Tier-A mass must equal the reported K_eff.
    assert learner.lambda_matrix_ is not None
    tier_a_rows = int(np.sum(np.max(learner.lambda_matrix_, axis=1) > 0.0105))
    assert tier_a_rows == learner.k_eff_

    # Large sparse repertoire (n_rem > 5000, max DF >= 10) switches to min_df = 2.
    big_df = np.ones(300, dtype=np.float64)
    big_df[:20] = 12.0
    mask_big, n_core_big, k_eff_big = compute_core_mask_and_k_eff(big_df, 6000, 10)
    assert n_core_big == 20 and int(np.sum(mask_big)) == 20
    assert k_eff_big == 2  # floor of min(2, K)
    empty_mask, empty_core, empty_k = compute_core_mask_and_k_eff(np.zeros(0), 100, 4)
    assert len(empty_mask) == 0 and empty_core == 0 and empty_k == 0


def test_network_profiles_consistent_between_evaluator_and_simulator() -> None:
    """Regression: GIN-2g-poor (and all shared profile names) must carry identical
    RTT / bandwidth numbers in schemas.STANDARD_EVALUATION_PARAMS and in
    simulator.get_network_profile (they previously disagreed for GIN-2g-poor)."""
    from mlfont import STANDARD_EVALUATION_PARAMS, get_network_profile

    by_name = {p.name: p for p in STANDARD_EVALUATION_PARAMS}
    for name in ("GIN-2g-poor", "GIN-2g", "GIN-3g", "residential-DSL", "residential-ADSL2+"):
        ev = by_name[name]
        bp = get_network_profile(name)
        assert bp.rtt_ms == ev.rtt_ms, name
        assert bp.bandwidth_bytes_per_ms == ev.bandwidth_bytes_per_ms, name
    # README numbers: bytes/ms * 8 = kbps
    assert by_name["GIN-2g"].bandwidth_bytes_per_ms * 8 == 100.0
    assert by_name["GIN-3g"].bandwidth_bytes_per_ms * 8 == 400.0
    assert by_name["residential-DSL"].bandwidth_bytes_per_ms * 8 == 2000.0
    assert by_name["residential-ADSL2+"].bandwidth_bytes_per_ms * 8 == 20000.0


def test_evaluator_serial_rtt_latency_model_hand_computed() -> None:
    """Pins the documented per-page latency model of FontLoadingEvaluator."""
    cps = list(range(0x41, 0x41 + 8))  # A..H
    strat = SlicingStrategy(subsets=[cps[:4], cps[4:6], cps[6:]], name="s3")
    # Page 1 touches slices 0 and 1 (3 covered glyphs); page 2 touches slice 1
    # (cached) and slice 2 (2 covered glyphs); page 3 has a codepoint outside the
    # strategy (missing) plus one in cached slice 0.
    strat_missing = SlicingStrategy(subsets=[cps[:4], cps[4:6]], name="s2_missing")
    samples = [{0x41, 0x42, 0x45}, {0x45, 0x47}, {0x41, 0x47}]
    ev = FontLoadingEvaluator(cps, samples)
    params = EvaluationParams(
        name="t", rtt_ms=100.0, bandwidth_bytes_per_ms=10.0,
        average_glyph_size_bytes=50.0, session_length=10,
    )
    res = ev.evaluate_strategy(strat, params)
    # page1: 2 new slices -> 200 ms RTT / 3 glyphs ; bytes = (4+2)*50 = 300 -> 30 ms / 3
    # page2: 1 new slice (2) -> 100 / 2 ; bytes = 2*50 = 100 -> 10 ms / 2
    # page3: everything cached -> 0
    assert np.isclose(res.rtt_per_glyph_ms.mean, (200 / 3 + 100 / 2 + 0.0) / 3)
    assert np.isclose(res.download_per_glyph_ms.mean, (30 / 3 + 10 / 2 + 0.0) / 3)
    assert np.isclose(
        res.total_per_glyph_ms.mean, res.rtt_per_glyph_ms.mean + res.download_per_glyph_ms.mean
    )
    assert res.num_missing_codepoints == 0

    res_m = ev.evaluate_strategy(strat_missing, params)
    # G (0x47) is not covered: counted as missing, excluded from the denominator.
    assert res_m.num_missing_codepoints == 1
    assert res_m.num_unassigned_font_codepoints == 2
    # page2: slice 1 cached, G missing -> 1 covered glyph, nothing new -> 0 ms
    assert np.isclose(res_m.rtt_per_glyph_ms.mean, (200 / 3 + 0.0 + 0.0) / 3)


def test_headless_chrome_fallback_never_fabricates_exact_match(tmp_path: Path, monkeypatch):
    """Regression: when Chrome emits no report, the benchmark must not report
    simulator_exact_slice_match=True (it previously copied the simulator's own
    prediction and compared it against itself)."""
    import json
    import shutil
    import subprocess
    import types

    from mlfont import SlicingStrategy
    from mlfont.simulator import ChromeBrowserPageResult, run_headless_chrome_benchmark

    cps = list(range(0x4E00, 0x4E00 + 40))
    strat = SlicingStrategy(
        subsets=[cps[:10], cps[10:20], cps[20:30], cps[30:]],
        name="range_compact_topics4",
    )
    bdir = tmp_path / "mlfont-notosanssc"
    (bdir / "woff2").mkdir(parents=True)
    strat.save_json(bdir / "strategy.json")
    (bdir / "config.json").write_text(
        json.dumps({"preset": "unknown_preset", "font_family": "Tiny Sliced", "n_topics": 4}),
        encoding="utf-8",
    )
    for i in range(4):
        (bdir / "woff2" / f"NotoSansSC-Regular_slice_{i:03d}.woff2").write_bytes(b"\0" * 10)

    from mlfont.simulator import WebPageSample

    page = WebPageSample(
        page_id="p", title="P", url="u", category="c", font_key="notosanssc",
        html_content="<p>" + "".join(chr(c) for c in cps[:12]) + "</p>",
    )

    monkeypatch.setattr(shutil, "which", lambda name: "/fake/chrome")

    def _fake_run(cmd, **kwargs):  # Chrome crashed / produced no DOM dump
        return types.SimpleNamespace(stdout="", stderr="boom", returncode=1)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    results = run_headless_chrome_benchmark(bdir, pages=[page], full_woff2_dir=tmp_path)
    assert results and all(isinstance(r, ChromeBrowserPageResult) for r in results)
    assert all(r.chrome_report_available is False for r in results)
    assert not any(r.simulator_exact_slice_match for r in results)

    # A genuine Chrome report that loads exactly the predicted rules -> exact match True
    def _good_run(cmd, **kwargs):
        html_file = [a for a in cmd if a.startswith("file://")][0][len("file://"):]
        n_rules = Path(html_file).read_text(encoding="utf-8").count("@font-face")
        # Only count the rules the simulator predicts for this strategy
        pred = {"single_slice": [0], "equal_bin_4": [0, 1], "range_compact_topics4": [0, 1]}
        sname = [k for k in pred if f"?s={k}&" in Path(html_file).read_text(encoding="utf-8")]
        idx = pred[sname[0]] if sname else list(range(min(1, n_rules)))
        rep = json.dumps({"local_decode_ms": 5.0, "requested_slice_indices": idx})
        return types.SimpleNamespace(
            stdout=f'<pre id="mlfont-chrome-report">{rep}</pre>', stderr="", returncode=0
        )

    monkeypatch.setattr(subprocess, "run", _good_run)
    # When full_woff2 is missing, single_slice must not substitute slice_000.woff2
    results_no_full = run_headless_chrome_benchmark(bdir, pages=[page], full_woff2_dir=tmp_path)
    single_no_full = [r for r in results_no_full if r.strategy_name == "single_slice"][0]
    assert single_no_full.chrome_report_available is False
    assert single_no_full.simulator_exact_slice_match is False

    # When full_woff2 is present, all strategies succeed and match
    (tmp_path / "NotoSansSC-Regular.woff2").write_bytes(b"\0" * 40)
    results_ok = run_headless_chrome_benchmark(bdir, pages=[page], full_woff2_dir=tmp_path)
    assert all(r.chrome_report_available for r in results_ok)
    assert all(r.simulator_exact_slice_match for r in results_ok)


def test_model_card_chrome_section_reports_actual_match_count(tmp_path: Path) -> None:
    """Regression: the model-card Chrome section must print the computed exact-match
    ratio (not a hard-coded 100%) and flag incomplete fetches."""
    import json

    from scripts.run_noto_experiments import _build_website_sim_section

    def _row(page, strat, match, n_req):
        return {
            "page_id": page, "page_title": page, "font_key": "notosanssc",
            "strategy_name": strat, "profile_name": "Cable-5Mbps",
            "chrome_local_decode_ms": 10.0, "network_transfer_ms": 100.0,
            "fonts_ready_ms": 110.0, "load_event_end_ms": 110.0,
            "css_transfer_bytes": 1000, "css_duration_ms": 30.0,
            "num_slices_requested_by_chrome": n_req, "requested_slice_indices": list(range(n_req)),
            "simulator_exact_slice_match": match, "woff2_transfer_bytes": 5000,
            "max_slice_duration_ms": 50.0, "wall_clock_font_window_ms": 60.0,
            "local_decode_speedup_vs_single_slice": 1.0, "speedup_vs_single_slice": 1.0,
            "bytes_saved_vs_single_slice_pct": 0.0,
        }

    report = {
        "fonts": {"notosanssc": {"summaries": [], "full_font_woff2_bytes": 1, "page_results": []}},
        "headless_chrome_benchmark": [
            _row("p1", "single_slice", True, 1),
            _row("p1", "range_compact_x", True, 3),
            _row("p2", "range_compact_x", False, 1),
        ],
    }
    rp = tmp_path / "sim.json"
    rp.write_text(json.dumps(report), encoding="utf-8")
    text = "\n".join(
        _build_website_sim_section(
            key="notosanssc", hf_name="mlfont-notosanssc", title="SC",
            strategy_name="range_compact_x", sim_report_path=rp,
        )
    )
    assert "2/3 = 66.7%" in text
    assert "100%" not in text
    assert "No*" in text
    assert "`p2` / `range_compact_x`" in text


def test_mlfont_js_syntax_and_documented_api() -> None:
    """Sanity-check the documented web/mlfont.js API surface and (when a JS parser
    is installed) that the file is syntactically valid ES5."""
    from scripts.run_noto_experiments import REPO_ROOT

    js = (REPO_ROOT / "web" / "mlfont.js").read_text(encoding="utf-8")
    for api in ("load: loadSlicedFont", "getMetrics: getMetrics", "printReport: printReport"):
        assert api in js
    assert "https://huggingface.co/wq2012" in js
    # Full syntax check when a JS parser is available (optional dependency).
    try:
        import pyjsparser  # type: ignore
    except ImportError:
        pyjsparser = None
    if pyjsparser is not None:
        pyjsparser.parse(js)


def test_round1_review_fixes(tmp_path: Path, monkeypatch) -> None:
    """Verify Round 1 review fixes across evaluator, train CLI, DNNTM, GA, and demo.html."""
    import json
    import sys
    from mlfont import (
        DnnTmConfig,
        DnnTmLearner,
        DocumentCorpusProcessor,
        EvaluationParams,
        FontLoadingEvaluator,
        FrequencySolver,
        RangeCompactConfig,
        RangeCompactGASolver,
        SlicingStrategy,
        WebsiteFontSimulator,
        export_pretrained_bundle,
        load_pretrained,
    )
    from scripts import train
    from scripts.run_noto_experiments import REPO_ROOT

    # 1. Evaluator partial trailing session utilization (4 samples, L=3 -> 2 sessions: 3 and 1)
    cps = list(range(0x41, 0x41 + 10))
    strat = SlicingStrategy(subsets=[cps[:5], cps[5:]], name="s2")
    samples = [
        {cps[0], cps[1]},
        {cps[0], cps[2]},
        {cps[0], cps[3]},
        {cps[5]},  # 4th sample is in trailing partial session (1 / 5 = 20%)
    ]
    ev = FontLoadingEvaluator(cps, samples)
    params_l3 = EvaluationParams(
        name="l3", rtt_ms=100.0, bandwidth_bytes_per_ms=10.0, session_length=3
    )
    res_l3 = ev.evaluate_strategy(strat, params_l3)
    # Session 1: 4 unique cps / 5 downloaded = 80%; Session 2 (trailing): 1 / 5 = 20% -> mean 50%
    assert np.isclose(res_l3.session_utilization_pct, 50.0)

    # 2. DnnTmLearner fitted attributes & empty active_cols short-circuit
    empty_corpus = DocumentCorpusProcessor(["xyz"])
    dnn_empty = DnnTmLearner(
        empty_corpus, list(range(0x4E00, 0x4E00 + 20)), DnnTmConfig(n_topics=3, training_steps=10)
    )
    strat_empty = dnn_empty.fit()
    assert dnn_empty.num_active_codepoints_ == 0
    assert dnn_empty.num_core_codepoints_ == 0
    assert dnn_empty.k_eff_ == 0
    assert dnn_empty.loss_history_ == []
    assert strat_empty.num_codepoints == 20

    active_corpus = DocumentCorpusProcessor(["一二三", "二三四", "五六七"])
    dnn_act = DnnTmLearner(
        active_corpus,
        [ord(c) for c in "一二三四五六七八九十"],
        DnnTmConfig(n_topics=2, training_steps=5),
    )
    dnn_act.fit()
    assert dnn_act.num_active_codepoints_ == 7
    assert dnn_act.num_core_codepoints_ == 7
    assert dnn_act.k_eff_ == 2

    # 3. RangeCompactGASolver.optimize guard against duplicate prefix_slices
    lam = np.ones((2, 6), dtype=np.float64)
    freq = np.array([100, 90, 10, 10, 10, 10, 10, 10], dtype=np.int64)
    all_8 = list(range(0x41, 0x41 + 8))
    ga_sol = RangeCompactGASolver(
        codepoints=all_8,
        topic_matrix=lam,
        config=RangeCompactConfig(max_iteration=5),
        frequency_array=freq,
        most_sizes=[2],
    )
    opt_strat = ga_sol.optimize(prefix_slices=[[0x41, 0x42]])
    assert opt_strat.num_codepoints == 8
    assert sum(len(s) for s in opt_strat.subsets) == 8

    # 4. FrequencySolver positional small codepoint list
    fs = FrequencySolver([0x41, 0x42, 0x43, 0x44])
    fs.fit(["AB", "ABC", "ABCD"])
    most_s, rem_s = fs.partition_most_frequent([2])
    assert len(most_s) == 1 and len(most_s[0]) == 2 and len(rem_s) == 2

    # 5. scripts/train.py CLI overrides when --preset is passed
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "codepoints.nam").write_text(
        "\n".join(f"0x{c:04X}" for c in range(0x41, 0x41 + 24)), encoding="utf-8"
    )
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps({"text": "ABCDEFGH"}) for _ in range(5)), encoding="utf-8"
    )
    out_train = tmp_path / "out_preset_override"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlfont-train",
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(out_train),
            "--preset",
            "lgc",
            "--algorithm",
            "online_lda",
            "--n_topics",
            "3",
            "--most_sizes",
            "4",
            "--normalization_method",
            "sinkhorn",
            "--seed",
            "77",
        ],
    )
    train.main()
    st_override = SlicingStrategy.load_json(out_train / "strategy.json")
    assert st_override.num_slices == 4  # 1 most slice (size 4) + 3 topic slices

    # 6. export_pretrained_bundle font_sliced.css url(woff2/...) & load_pretrained dna_assignment
    b_out = tmp_path / "bundle_test"
    export_pretrained_bundle(
        output_dir=b_out,
        strategy=st_override,
        lambda_matrix=np.ones((3, 20), dtype=np.float32),
        dnntm_lambda_matrix=np.ones((3, 20), dtype=np.float32),
        olda_strategy=st_override,
        dnntm_strategy=st_override,
        slice_font=False,
    )
    css_root = (b_out / "font_sliced.css").read_text(encoding="utf-8")
    assert "url(woff2/font_slice_000.woff2)" in css_root
    assert (b_out / "mlfont.js").exists()
    loaded_b = load_pretrained(b_out)
    assert loaded_b.dna_assignment is not None
    assert loaded_b.dnntm_lambda_matrix is not None
    assert loaded_b.olda_strategy is not None
    assert loaded_b.dnntm_strategy is not None

    # 7. WebsiteFontSimulator.from_pretrained LGC high_freq_slice_count == 0 & demo.html checks
    sim_lgc = WebsiteFontSimulator.from_pretrained(
        REPO_ROOT / "pretrained_models" / "mlfont-notosans-lgc"
    )
    assert sim_lgc.high_freq_slice_count == 0
    demo_text = (REPO_ROOT / "web" / "demo.html").read_text(encoding="utf-8")
    assert "repertoireDeltas" in demo_text and "gaSliceB64" in demo_text
    assert "uniqueChars / baseChars" not in demo_text
    assert "* 0.18" not in demo_text
