import re
import math
from urllib.parse import urlparse
from collections import Counter

# =========================
# ENTROPY FUNCTION
# =========================
def entropy(text):
    if not text:
        return 0
    prob = [count / len(text) for count in Counter(text).values()]
    return -sum(p * math.log2(p) for p in prob)


# =========================
# TRUSTED DOMAIN
# =========================
trusted_domains = [
    "google.com", "youtube.com",
    "facebook.com", "instagram.com",
    "shopee.co.id", "tokopedia.com"
]

# =========================
# PHISHING KEYWORDS
# =========================
phishing_keywords = [
    "login", "verify", "update", "secure",
    "account", "bank", "confirm", "password"
]

# =========================
# FEATURE EXTRACTION (18 FEATURES)
# =========================
def extract_features(url):
    try:
        parsed = urlparse(url)

        # 🔥 FIX: handle URL tanpa http/https
        if not parsed.netloc:
            parsed = urlparse("http://" + url)

        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()

        url_length = len(url)

        # =========================
        # BASIC FEATURES
        # =========================
        has_ip_address = 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', domain) else 0

        dot_count = domain.count('.')

        https_flag = 1 if parsed.scheme == "https" else 0

        url_entropy = entropy(url)

        tokens = re.split(r'[./\-?=&]', url)
        token_count = len([t for t in tokens if t])

        parts = domain.split('.')
        subdomain_count = max(len(parts) - 2, 0)

        query_param_count = len(query.split('&')) if query else 0

        tld = parts[-1] if parts else ""
        tld_length = len(tld)

        path_length = len(path)

        has_hyphen_in_domain = 1 if "-" in domain else 0

        number_of_digits = sum(c.isdigit() for c in url)

        # =========================
        # ADVANCED FEATURES
        # =========================
        popular_tlds = ["com", "org", "net", "id", "co", "io"]
        tld_popularity = 1 if tld in popular_tlds else 0

        suspicious_file_extension = 1 if path.endswith(
            (".php", ".exe", ".zip", ".js")
        ) else 0

        domain_name_length = len(domain)

        percentage_numeric_chars = (
            number_of_digits / url_length if url_length > 0 else 0
        )

        url_lower = url.lower()

        has_phishing_keyword = 1 if any(
            k in url_lower for k in phishing_keywords
        ) else 0

        is_trusted_domain = 1 if any(
            domain.endswith(td) for td in trusted_domains
        ) else 0

        # =========================
        # FINAL VECTOR (18 FEATURES)
        # =========================
        return [
            url_length,
            has_ip_address,
            dot_count,
            https_flag,
            url_entropy,
            token_count,
            subdomain_count,
            query_param_count,
            tld_length,
            path_length,
            has_hyphen_in_domain,
            number_of_digits,
            tld_popularity,
            suspicious_file_extension,
            domain_name_length,
            percentage_numeric_chars,
            has_phishing_keyword,
            is_trusted_domain
        ]

    except Exception as e:
        print("ERROR feature extraction:", e)
        return [0] * 18  # fallback biar app nggak crash