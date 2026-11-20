"""Adapted directly from the provided standalone transform experiments.
20 ms/320-sample frame interface, no experiment file I/O or codec tests.
The PCM16 network transport is lossy for scrambled coefficients.
"""
import hashlib
import hmac
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.fft import dct
FRAME = 320

def configure_frame_size(size):
    """Set the 20 ms voice payload size (fixed for the process lifetime)."""
    global FRAME
    size = int(size)
    if size <= 0 or size % 64:
        raise ValueError('Voice frame size must be positive and divisible by 64 (LWT 6 levels)')
    FRAME = size

FRFT_ORDER = .5
LWT_LEVELS = 6
SQRT2 = np.sqrt(2.)
SIGN_EPS = 1e-10
PHASE_EPS = 1e-10
BIT_OPERATIONS = frozenset(('sbox', 'rot-xor'))
PAIRED_OPERATIONS = ('phase-offset','amplitude-permutation','amplitude-phase')
ENGINE_VERSION = 'network-frame-v2'


def _fractional_dft(z, order):
    """Унитарная спектральная дробная степень дискретного преобразования Фурье.

    F = FFT(norm='ortho'), F**4 = I. Спектральные проекторы на собственные
    значения (-1j)**k обеспечивают точную обратимость F**a * F**(-a) = I.
    Это *одна конкретная* дискретная реализация FRFT, не утверждается, что
    она совпадает с реализацией FRFT из внешнего crypto/adapter.py.
    """
    z = np.asarray(z, dtype=np.complex128)
    projections = np.zeros_like(z)
    f = [z]
    for _ in range(3):
        f.append(np.fft.fft(f[-1], axis=-1, norm='ortho'))
    result = np.zeros_like(z)
    for k in range(4):
        eigenvalue = (-1j) ** k
        # P_k = (1/4) sum_m (lambda_k**(-m)) F**m
        projection = sum((eigenvalue ** (-m)) * f[m] for m in range(4)) / 4.0
        result += np.exp(-0.5j * np.pi * order * k) * projection
    return result


def frft(blocks):
    """Дискретная FRFT: упаковка 320 отсчётов в 160 комплексных пар."""
    x = np.asarray(blocks, dtype=np.float64)
    if x.shape[-1] != FRAME:
        raise ValueError(f'Ожидалось {FRAME} отсчётов в блоке')
    z = x[..., ::2] + 1j * x[..., 1::2]
    transformed = _fractional_dft(z, FRFT_ORDER)
    out = np.empty(x.shape, dtype=np.float64)
    out[..., ::2] = transformed.real
    out[..., 1::2] = transformed.imag
    return out


def ifrft(coefficients):
    """Обратная FRFT с порядком -FRFT_ORDER."""
    x = np.asarray(coefficients, dtype=np.float64)
    if x.shape[-1] != FRAME:
        raise ValueError(f'Ожидалось {FRAME} компонент FRFT')
    z = x[..., ::2] + 1j * x[..., 1::2]
    restored = _fractional_dft(z, -FRFT_ORDER)
    out = np.empty(x.shape, dtype=np.float64)
    out[..., ::2] = restored.real
    out[..., 1::2] = restored.imag
    return out


def fresnel(blocks):
    """Дискретное преобразование Френеля, 320 вещественных -> 160 комплексных.

    Соседние отсчёты упаковываются в комплексные пары. Используется
    унитарный оператор chirp * DFT * chirp, соответствующий дискретной
    форме интеграла Френеля. Результат снова упаковывается в 320
    вещественных компонент (Re, Im), чтобы применять те же 7 методов.
    """
    x = np.asarray(blocks, dtype=np.float64)
    if x.shape[-1] != FRAME:
        raise ValueError(f'Ожидалось {FRAME} отсчётов в блоке')
    z = x[..., ::2] + 1j * x[..., 1::2]
    size = FRAME // 2
    chirp = np.exp(1j * np.pi * np.arange(size, dtype=np.float64)**2 / size)
    transformed = np.fft.fft(z * chirp, axis=-1, norm='ortho') * chirp
    out = np.empty(x.shape, dtype=np.float64)
    out[..., ::2] = transformed.real
    out[..., 1::2] = transformed.imag
    return out


def ifresnel(coefficients):
    """Точное обратное преобразование к fresnel() с точностью float64."""
    x = np.asarray(coefficients, dtype=np.float64)
    if x.shape[-1] != FRAME:
        raise ValueError(f'Ожидалось {FRAME} компонент Fresnel')
    z = x[..., ::2] + 1j * x[..., 1::2]
    size = FRAME // 2
    chirp = np.exp(1j * np.pi * np.arange(size, dtype=np.float64)**2 / size)
    restored = np.fft.ifft(z / chirp, axis=-1, norm='ortho') / chirp
    out = np.empty(x.shape, dtype=np.float64)
    out[..., ::2] = restored.real
    out[..., 1::2] = restored.imag
    return out


def _lifting_forward(values):
    values = np.asarray(values, dtype=np.float64)
    even = values[..., ::2].copy()
    odd = values[..., 1::2].copy()
    # Predict: odd_i -= (even_i + even_{i+1})/2.
    right = np.concatenate((even[..., 1:], even[..., -1:]), axis=-1)
    detail = odd - 0.5 * (even + right)
    # Update: even_i += (detail_{i-1} + detail_i)/4.
    left = np.concatenate((detail[..., :1], detail[..., :-1]), axis=-1)
    approx = even + 0.25 * (left + detail)
    return approx, detail


def _lifting_inverse(approx, detail):
    approx = np.asarray(approx, dtype=np.float64)
    detail = np.asarray(detail, dtype=np.float64)
    left = np.concatenate((detail[..., :1], detail[..., :-1]), axis=-1)
    even = approx - 0.25 * (left + detail)
    right = np.concatenate((even[..., 1:], even[..., -1:]), axis=-1)
    odd = detail + 0.5 * (even + right)
    out = np.empty(approx.shape[:-1] + (approx.shape[-1] * 2,), dtype=np.float64)
    out[..., ::2], out[..., 1::2] = even, odd
    return out


def lwt(blocks):
    """6 уровней lifting CDF 5/3, упаковка [a6,d6,d5,...,d1]."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.shape[-1] != FRAME:
        raise ValueError(f'Для LWT требуется блок из {FRAME} отсчётов')
    approx = blocks
    details = []
    for _ in range(LWT_LEVELS):
        approx, detail = _lifting_forward(approx)
        details.append(detail)
    return np.concatenate([approx] + details[::-1], axis=-1)


def ilwt(coefficients):
    """Обратный CDF 5/3 lifting: восстановление всех 320 отсчётов."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[-1] != FRAME:
        raise ValueError(f'Для ILWT требуется {FRAME} коэффициентов')
    size = FRAME // (2 ** LWT_LEVELS)
    approx = coefficients[..., :size]
    offset = size
    for _ in range(LWT_LEVELS):
        detail = coefficients[..., offset:offset + size]
        approx = _lifting_inverse(approx, detail)
        offset += size
        size *= 2
    return approx


def dtmt_basis(size=None):
    if size is None:
        size = FRAME
    """Строка T[k, :] — нормированный полином Чебышёва порядка k на 0..N-1.

    Та же матрица Якоби и тот же выбор знаков, что в crypto/dtmt.py проекта.
    T @ x даёт моменты, T.T @ c возвращает отсчёты.
    """
    if size < 2:
        raise ValueError('Для DTMT требуется минимум два отсчёта')
    order = np.arange(1, size, dtype=np.float64)
    diagonal = np.full(size, (size - 1) / 2.0)
    side = 0.5 * order * np.sqrt((size**2 - order**2) / (4 * order**2 - 1))
    nodes, basis = eigh_tridiagonal(diagonal, side)
    if np.max(np.abs(nodes - np.arange(size))) > 1e-8:
        raise ArithmeticError('Узлы базиса DTMT расходятся с сеткой 0..N-1')
    # Знак собственного вектора произволен. Нулевой момент имеет положительный вес.
    basis *= np.where(basis[0, :] < 0, -1.0, 1.0)
    basis.setflags(write=False)
    return basis


def dtmt(blocks):
    """Анализ одного блока или массива блоков, по последней оси."""
    return np.asarray(blocks, dtype=np.float64) @ dtmt_basis(np.asarray(blocks).shape[-1]).T


def idtmt(moments):
    """Синтез одного блока или массива блоков по всем 320 моментам."""
    return np.asarray(moments, dtype=np.float64) @ dtmt_basis(np.asarray(moments).shape[-1])


def pack_spectrum(coeff):
    """N реальных компонент: DC, Nyquist, sqrt(2)*Re/Im внутренних бинов.

    Сумма квадратов этого вектора равна энергии временного блока.
    Допускаются несколько спектров: последняя ось считается частотной.
    """
    coeff = np.asarray(coeff, dtype=np.complex128)
    if coeff.ndim < 1 or coeff.shape[-1] < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError('Некорректный rFFT-спектр')
    if np.any(coeff[..., [0, -1]].imag != 0):
        raise ValueError('Нельзя терять мнимую часть DC или Nyquist')
    count = 2 * (coeff.shape[-1] - 1)
    result = np.empty(coeff.shape[:-1] + (count,), dtype=np.float64)
    result[..., 0] = coeff[..., 0].real
    result[..., 1] = coeff[..., -1].real
    result[..., 2::2] = SQRT2 * coeff[..., 1:-1].real
    result[..., 3::2] = SQRT2 * coeff[..., 1:-1].imag
    return result


def unpack_spectrum(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError('Нужен чётный вектор компонент rFFT')
    if not np.all(np.isfinite(values)):
        raise ValueError('Нечисловые компоненты rFFT')
    result = np.empty(values.shape[:-1] + (values.shape[-1] // 2 + 1,), dtype=np.complex128)
    result[..., 0] = values[..., 0]
    result[..., -1] = values[..., 1]
    result[..., 1:-1] = (values[..., 2::2] + 1j * values[..., 3::2]) / SQRT2
    return result


def modify_components(values, operation, key, frame_index, *, decrypt=False,
                      file_number=0, quant_scale=None, allow_corrupt=False, counters=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Некорректные вещественные компоненты rFFT')
    rng = frame_rng_stft(key, file_number, operation, frame_index)
    size = len(values)
    if operation == 'orthomix':
        indices = rng.permutation(size)
        pairs = size // 2
        first, second = indices[:2*pairs:2], indices[1:2*pairs:2]
        angles = rng.uniform(-np.pi, np.pi, size=pairs)
        if decrypt:
            angles = -angles
        c, s = np.cos(angles), np.sin(angles)
        a, b = values[first], values[second]
        result = values.copy()
        result[first] = c*a - s*b
        result[second] = s*a + c*b
        return result
    if operation in BIT_OPERATIONS:
        if quant_scale is None or not np.isfinite(quant_scale) or quant_scale <= 0:
            raise ValueError('Нужен положительный quant_scale из метаданных файла')
        scaled = values * quant_scale
        if not np.all(np.isfinite(scaled)):
            raise ValueError('Переполнение при квантовании rFFT-компонент')
        rounded = np.rint(scaled)
        outside = (rounded < -32768) | (rounded > 32767)
        count = int(np.count_nonzero(outside))
        if count:
            if not (decrypt and allow_corrupt):
                raise ValueError(f'{operation}: {count} коэффициентов вне int16')
            # Только повреждённый канал: насыщение для продолжения эксперимента.
            # В JSON видно число таких слов; это не восстановление информации.
            rounded = np.clip(rounded, -32768, 32767)
        if counters is not None:
            counters['out_of_range_words'] += count
        words = rounded.astype(np.int16).view(np.uint16)
        modified = modify_words(words, rng, operation, decrypt)
        return modified.view(np.int16).astype(np.float64) / quant_scale
    raise ValueError(f'Неизвестная операция: {operation}')


def modify_spectrum(coeff, operation, key, frame_index, *, decrypt=False,
                    file_number=0, quant_scale=None, phase_eps=PHASE_EPS,
                    allow_corrupt=False, counters=None):
    """Операции над комплексным спектром реального блока.

    shuffle и полярные операции работают на внутренних комплексных бинах.
    У DC/Nyquist нет произвольной комплексной фазы: они остаются реальными.
    orthomix и битовые операции используют все N вещественных компонент.
    """
    coeff = np.asarray(coeff, dtype=np.complex128)
    if coeff.ndim != 1 or len(coeff) < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError('Некорректный спектр блока')
    if np.any(coeff[[0, -1]].imag != 0):
        raise ValueError('DC и Nyquist должны оставаться вещественными')

    if operation == 'invert':
        # Инверсия порядка частот. DC и Nyquist меняются местами, оба вещественные.
        return coeff[::-1].copy()
    if operation == 'orthomix' or operation in BIT_OPERATIONS:
        changed = modify_components(
            pack_spectrum(coeff), operation, key, frame_index, decrypt=decrypt,
            file_number=file_number, quant_scale=quant_scale,
            allow_corrupt=allow_corrupt, counters=counters)
        return unpack_spectrum(changed)

    rng = frame_rng_stft(key, file_number, operation, frame_index)
    if operation == 'signflip':
        signs = 2.0 * rng.integers(0, 2, size=len(coeff)) - 1.0
        return coeff * signs

    result = coeff.copy()
    interior = coeff[1:-1]
    count = len(interior)
    if operation == 'shuffle':
        permutation = rng.permutation(count)
        result[1:-1] = interior[np.argsort(permutation) if decrypt else permutation]
        return result
    if operation == 'phase-offset':
        angles = rng.uniform(-np.pi, np.pi, size=count)
        result[1:-1] = interior * np.exp(1j * (-angles if decrypt else angles))
        return result
    if operation in ('amplitude-permutation', 'amplitude-phase'):
        if not np.isfinite(phase_eps) or phase_eps <= 0:
            raise ValueError('phase_eps должен быть положительным конечным')
        permutation = rng.permutation(count)
        angles = (rng.uniform(-np.pi, np.pi, size=count)
                  if operation == 'amplitude-phase' else np.zeros(count))
        if decrypt:
            radius = np.maximum(np.abs(interior)[np.argsort(permutation)] - phase_eps, 0.0)
            phase = np.angle(interior) - angles
        else:
            radius = np.abs(interior)[permutation] + phase_eps
            phase = np.angle(interior) + angles
        result[1:-1] = radius * np.exp(1j * phase)
        return result
    raise ValueError(f'Неизвестная операция: {operation}')


def frame_rng(key, file_number, operation, frame_index):
    """Воспроизводимые параметры эксперимента, разделённые по файлу/методу/блоку.

    PCG64 здесь используется как генератор параметров исследования.
    Это не реализация стандартного криптографического протокола.
    """
    context = f'{ENGINE_VERSION}|{file_number}|{operation}|{frame_index}'.encode('utf-8')
    seed = int.from_bytes(hmac.new(key, context, hashlib.sha256).digest(), 'big')
    return np.random.Generator(np.random.PCG64(seed))


def _gf256_mul(a, b):
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 255
        b >>= 1
    return result


def _gf256_pow(x, exponent):
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf256_mul(result, x)
        x = _gf256_mul(x, x)
        exponent >>= 1
    return result


def _make_aes_sbox():
    table = np.empty(256, dtype=np.uint8)
    for x in range(256):
        inv = _gf256_pow(x, 254) if x else 0
        value = inv
        for shift in range(1, 5):
            value ^= ((inv << shift) | (inv >> (8 - shift))) & 255
        table[x] = value ^ 0x63
    return table


def rotate_left16(words, shifts):
    values = np.asarray(words, dtype=np.uint32)
    shifts = np.asarray(shifts, dtype=np.uint32)
    return (((values << shifts) | (values >> (16 - shifts))) & 65535).astype(np.uint16)


def rotate_right16(words, shifts):
    values = np.asarray(words, dtype=np.uint32)
    shifts = np.asarray(shifts, dtype=np.uint32)
    return (((values >> shifts) | (values << (16 - shifts))) & 65535).astype(np.uint16)


def modify_words(words, rng, operation, decrypt=False):
    """Точные перестановки 16-битных слов. Без float, FFT и квантования."""
    words = np.asarray(words, dtype=np.uint16)
    size = len(words)
    if operation == 'rot-xor':
        mask = rng.integers(0, 65536, size=size, dtype=np.uint16)
        shifts = rng.integers(1, 16, size=size, dtype=np.uint16)
        iv = np.uint16(rng.integers(0, 65536))
        if decrypt:
            previous = np.empty_like(words)
            previous[0] = iv
            previous[1:] = words[:-1]
            return rotate_right16(words ^ previous, shifts) ^ mask
        rotated = rotate_left16(words ^ mask, shifts)
        return np.bitwise_xor.accumulate(rotated) ^ iv
    if operation == 'sbox':
        first = rng.integers(0, 256, size=(2, size), dtype=np.uint8)
        second = rng.integers(0, 256, size=(2, size), dtype=np.uint8)
        low = (words & 255).astype(np.uint8)
        high = (words >> 8).astype(np.uint8)
        if decrypt:
            low = AES_SBOX_INV[low ^ second[0]] ^ first[0]
            high = AES_SBOX_INV[high ^ second[1]] ^ first[1]
        else:
            low = AES_SBOX[low ^ first[0]] ^ second[0]
            high = AES_SBOX[high ^ first[1]] ^ second[1]
        return low.astype(np.uint16) | (high.astype(np.uint16) << 8)
    raise ValueError(f'Не битовая операция: {operation}')


def native_rng(key, frame_index, domain):
    """Параметры прежних методов из nirs_timepack_v2, включая период 64 блока."""
    material = (b'NIRS_EXPERIMENTAL_V1\0' + key + b'\0FresnelCipher\0'
                + domain.encode('ascii') + b'\0' + (int(frame_index) % 64).to_bytes(8, 'big'))
    digest = hashlib.sha512(material).digest()
    seed = np.random.SeedSequence(np.frombuffer(digest, dtype='<u4'), pool_size=8)
    return np.random.Generator(np.random.PCG64(seed))


def sign_mask(key, frame_index, count):
    return native_rng(key, frame_index, 'signflip').choice([-1.0, 1.0], size=count)


def paired_operation(values, operation, rng, decrypt, phase_eps):
    """z[j] = c[2j] + i*c[2j+1]. Угол относится к этой паре компонент коэффициентов."""
    if values.size % 2:
        raise ValueError('Для комплексных пар требуется чётное число компонент коэффициентов')
    pairs = values[::2] + 1j * values[1::2]
    if operation == 'phase-offset':
        angles = rng.uniform(-np.pi, np.pi, size=pairs.size)
        changed = pairs * np.exp(1j * (-angles if decrypt else angles))
    else:
        permutation = rng.permutation(pairs.size)
        phase = np.angle(pairs)
        if operation == 'amplitude-phase':
            angles = rng.uniform(-np.pi, np.pi, size=pairs.size)
            phase = phase - angles if decrypt else phase + angles
        if decrypt:
            radius = np.maximum(np.abs(pairs)[np.argsort(permutation)] - phase_eps, 0.0)
        else:
            radius = np.abs(pairs)[permutation] + phase_eps
        changed = radius * np.exp(1j * phase)
    result = np.empty_like(values)
    result[::2], result[1::2] = changed.real, changed.imag
    return result


def bit_operation(values, operation, rng, decrypt, scale, allow_corrupt, counters):
    """Равномерное квантование компонент коэффициентов и обратимая операция на 16-битных словах."""
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Нужен положительный масштаб квантования из JSON')
    rounded = np.rint(values * scale)
    if not np.all(np.isfinite(rounded)):
        raise ValueError('Переполнение при квантовании коэффициентов FRFT')
    outside = (rounded < -32768) | (rounded > 32767)
    count = int(np.count_nonzero(outside))
    if count:
        if not (decrypt and allow_corrupt):
            raise ValueError(f'{operation}: {count} слов вне диапазона int16')
        # Принимаем и учитываем выход за диапазон после кодека с потерями.
        rounded = np.clip(rounded, -32768, 32767)
    if counters is not None:
        counters['out_of_range_words'] += count
    words = rounded.astype(np.int16).view(np.uint16)
    changed = modify_words(words, rng, operation, decrypt)
    return changed.view(np.int16).astype(np.float64) / scale


def modify_coefficients(values, operation, key, frame_index, *, file_number=0,
                        decrypt=False, quant_scale=None, sign_eps=SIGN_EPS,
                        phase_eps=PHASE_EPS, allow_corrupt=False, counters=None):
    """Прямая или обратная операция над вектором вещественных компонент коэффициентов блока."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Нужен вектор конечных вещественных компонент коэффициентов')
    count = values.size
    if operation == 'invert':
        return values[::-1].copy()
    if operation == 'signflip':
        return values * sign_mask(key, frame_index, count)
    if operation == 'shuffle':
        permutation = native_rng(key, frame_index, 'shuffle').permutation(count)
        return values[np.argsort(permutation) if decrypt else permutation]
    if operation == 'orthomix':
        rng = native_rng(key, frame_index, 'orthomix')
        permutation = rng.permutation(count)
        pairs = permutation[:2 * (count // 2)].reshape(-1, 2)
        left, right = pairs.T
        angles = rng.uniform(-np.pi, np.pi, size=len(pairs))
        cosine = np.cos(angles)
        sine = np.sin(-angles if decrypt else angles)
        result = values.copy()
        result[left] = cosine * values[left] - sine * values[right]
        result[right] = sine * values[left] + cosine * values[right]
        return result
    if operation == 'amplitude-sign':
        if not np.isfinite(sign_eps) or sign_eps <= 0:
            raise ValueError('sign_eps должен быть положительным конечным')
        permutation = native_rng(key, frame_index, 'magnitude').permutation(count)
        mask = sign_mask(key, frame_index, count)
        # Нулю назначаем знак +1: иначе sign(0) стёр бы чужой модуль.
        signs = np.where(values < 0, -1.0, 1.0)
        if decrypt:
            radius = np.maximum(np.abs(values)[np.argsort(permutation)] - sign_eps, 0.0)
        else:
            radius = np.abs(values)[permutation] + sign_eps
        return signs * mask * radius
    rng = frame_rng(key, file_number, operation, frame_index)
    if operation in BIT_OPERATIONS:
        return bit_operation(values, operation, rng, decrypt, quant_scale, allow_corrupt, counters)
    if operation in PAIRED_OPERATIONS:
        if not np.isfinite(phase_eps) or phase_eps <= 0:
            raise ValueError('phase_eps должен быть положительным конечным')
        return paired_operation(values, operation, rng, decrypt, phase_eps)
    raise ValueError(f'Неизвестная операция: {operation}')


def frame_rng_stft(key, file_number, operation, frame_index):
    """Воспроизводимые параметры эксперимента, разделённые по файлу/методу/блоку.

    PCG64 здесь используется как генератор параметров исследования.
    Это не реализация стандартного криптографического протокола.
    """
    context = f'{ENGINE_VERSION}|{file_number}|{operation}|{frame_index}'.encode('utf-8')
    seed = int.from_bytes(hmac.new(key, context, hashlib.sha256).digest(), 'big')
    return np.random.Generator(np.random.PCG64(seed))


AES_SBOX = _make_aes_sbox()
AES_SBOX_INV = np.empty(256, dtype=np.uint8)
AES_SBOX_INV[AES_SBOX] = np.arange(256, dtype=np.uint8)



TRANSFORMS = ('stft', 'dct4', 'mdct-no-overlap', 'frft','fresnel','lwt','dtmt')
ALL_OPERATIONS = ('invert','shuffle','signflip','orthomix','amplitude-sign',
                  'phase-offset','amplitude-permutation','amplitude-phase',
                  'adaptive-orthomix','sbox','rot-xor','sbox-16','rot-xor-16',
                  'sbox-16-float-codec','rot-xor-16-float-codec')


def forward(kind, frame):
    x = np.asarray(frame, dtype=np.float64)
    if kind == 'stft':
        return np.fft.rfft(x, norm='ortho')
    if kind in ('dct4','mdct-no-overlap'):
        return dct(x, type=4, norm='ortho')
    if kind == 'frft': return frft(x)
    if kind == 'fresnel': return fresnel(x)
    if kind == 'lwt': return lwt(x)
    if kind == 'dtmt': return dtmt(x)
    raise ValueError(f'Unknown transform: {kind}')


def inverse(kind, coeff):
    if kind == 'stft':
        return np.fft.irfft(coeff, n=2 * (len(coeff) - 1), norm='ortho')
    if kind in ('dct4','mdct-no-overlap'):
        return dct(coeff, type=4, norm='ortho')
    if kind == 'frft': return ifrft(coeff)
    if kind == 'fresnel': return ifresnel(coeff)
    if kind == 'lwt': return ilwt(coeff)
    if kind == 'dtmt': return idtmt(coeff)
    raise ValueError(f'Unknown transform: {kind}')


def apply_operation(kind, coeff, operation, key, index, decrypt=False, quant_scale=1024.):
    effective_index = int(index) % 64
    if kind == 'stft':
        if operation == 'amplitude-sign':
            v = pack_spectrum(coeff)
            v = modify_coefficients(v, operation, key, effective_index,
                                    decrypt=decrypt, quant_scale=quant_scale)
            return unpack_spectrum(v)
        return modify_spectrum(coeff, operation, key, effective_index,
                               decrypt=decrypt, quant_scale=quant_scale)
    return modify_coefficients(coeff, operation, key, effective_index,
                               decrypt=decrypt, quant_scale=quant_scale)


BIT_ALIASES = {'sbox-16': 'sbox', 'rot-xor-16': 'rot-xor',
               'sbox-16-float-codec': 'sbox',
               'rot-xor-16-float-codec': 'rot-xor'}


def apply_adaptive(values, key, index, decrypt):
    """Givens rotations with S-box-derived smooth radius-dependent angle.
    Adapted from stft_n_2_.py / mdct_rot_xor16_float64_final_2.py.
    This is NOT S-box substitution of the coefficients.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = frame_rng(key, 0, 'adaptive-orthomix', int(index) % 64)
    count = len(values)
    indices = rng.permutation(count)
    left, right = indices[:2*(count//2):2], indices[1:2*(count//2):2]
    a,b=values[left].copy(), values[right].copy()
    # Original code scales coefficients by COEFFICIENT_SCALE=32768 before
    # computing the relative radius, thus equivalent to radius / 8192.
    radius=np.hypot(a,b)
    offset=int(rng.integers(0,256))
    position=np.mod(radius/8192. + offset,256.)
    low=np.floor(position).astype(np.int64)
    frac=position-low
    blend=frac*frac*(3.-2.*frac)
    substitute=(1.-blend)*AES_SBOX[low].astype(float)+blend*AES_SBOX[(low+1)%256].astype(float)
    theta=2.3*(2.*substitute/255.-1.)
    c=np.cos(theta); t=np.sin(theta)
    out=values.copy()
    if decrypt:
        out[left]=c*a+t*b;out[right]=-t*a+c*b
    else:
        out[left]=c*a-t*b;out[right]=t*a+c*b
    return out


_original_apply_operation = apply_operation


def apply_operation(kind, coeff, operation, key, index, decrypt=False, quant_scale=1024.):
    if operation == 'adaptive-orthomix':
        if kind == 'stft':
            return unpack_spectrum(apply_adaptive(pack_spectrum(coeff),key,index,decrypt))
        return apply_adaptive(coeff,key,index,decrypt)
    return _original_apply_operation(kind, coeff, BIT_ALIASES.get(operation,operation),
                                     key, index, decrypt, quant_scale)
