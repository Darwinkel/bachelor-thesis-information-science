"""Script to iterate over the Tranco list,
 do tests, and write the results to a database"""
import concurrent.futures
import csv
import socket
import ssl
import subprocess
import sys
import time
from string import Template
from urllib.parse import urlsplit

from requests.structures import CaseInsensitiveDict

MAX_THREADS = 500
TIMEOUT = 2
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:105.0) Gecko/20100101 Firefox/105.0"

FILENAME = f"batch_{time.time()}.tsv"
DUMP = False

# Very basic HEAD request to retrieve canonical locations
TEST_CASE_0 = Template(
    f"HEAD / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"\r\n"
)

# Opportunistic / assumed HTTP/2 through curl
TEST_CASE_1 = None
TEST_CASE_2 = None

# HTTP/0.9
TEST_CASE_3 = Template("GET /\r\n\r\n")

# HTTP/1.0
TEST_CASE_4 = Template(f"GET / HTTP/1.0\r\nUser-Agent: {USER_AGENT}\r\n\r\n")

# HTTP/1.0 with Host disambiguation
TEST_CASE_5 = Template(
    f"GET / HTTP/1.0\r\nHost: $hostname\r\nUser-Agent: {USER_AGENT}\r\n\r\n"
)

# HEAD request with unusual newline markers
TEST_CASE_6 = Template(
    f"HEAD / HTTP/1.1\n"
    f"Host: $hostname\n"
    f"User-Agent: {USER_AGENT}\n"
    f"Accept: */*\n"
    f"\n"
)

# HEAD request with unusual newline markers
# NOTE: almost always bad request
TEST_CASE_7 = Template(
    f"HEAD / HTTP/1.1\r"
    f"Host: $hostname\r"
    f"User-Agent: {USER_AGENT}\r"
    f"Accept: */*\r"
    f"\r"
)

# HEAD request on non-existent HTTP/4.0 protocol
TEST_CASE_8 = Template(
    f"HEAD / HTTP/4.0\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"\r\n"
)

# SPOCK request
TEST_CASE_9 = Template(
    f"SPOCK / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"\r\n"
)

# Case-insensitive HEAD request
# NOTE: almost always bad request
TEST_CASE_10 = Template(
    f"HEad / httP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"\r\n"
)

# OPTIONS request to the root
TEST_CASE_11 = Template(
    f"OPTIONS / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"\r\n"
)

# Multipart GET
TEST_CASE_12 = Template(
    f"HEAD / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"Range: bytes=0-4, 6-10\r\n"
    f"\r\n"
)

# Conditional Modified GET in the future
TEST_CASE_13 = Template(
    f"HEAD / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"If-Modified-Since: Wed, 21 Oct 2029 07:28:01 GMT\r\n"
    f"\r\n"
)

# Conditional Unmodified GET in the past
TEST_CASE_14 = Template(
    f"HEAD / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: */*\r\n"
    f"If-Unmodified-Since: Wed, 21 Oct 1985 07:28:01 GMT\r\n"
    f"\r\n"
)

# Request root as unsafe format
TEST_CASE_15 = Template(
    f"HEAD / HTTP/1.1\r\n"
    f"Host: $hostname\r\n"
    f"User-Agent: {USER_AGENT}\r\n"
    f"Accept: application/javascript;text/css;q=0.2\r\n"
    f"\r\n"
)


SOCKET_TESTS = [
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
    TEST_CASE_12,
    TEST_CASE_13,
    TEST_CASE_14,
    TEST_CASE_15,
]

global_ssl_context = ssl.create_default_context()
global_ssl_context.check_hostname = False
global_ssl_context.verify_mode = ssl.CERT_NONE
socket.setdefaulttimeout(TIMEOUT)


class TestResults:
    """Data structure to run and represent tests"""

    def __init__(self, rank: int, domain: str) -> None:

        self.ip80 = ""
        self.ip443 = ""
        self.rank = rank
        self.domain = domain
        self.reachable80 = True
        self.reachable443 = True

        self.http80_amount_of_redirects = 0
        self.http80_hostname = domain
        self.http80_test_case_results: list[tuple[str, CaseInsensitiveDict[str]]] = []

        self.https443_amount_of_redirects = 0
        self.https443_hostname = domain
        self.https443_test_case_results: list[tuple[str, CaseInsensitiveDict[str]]] = []

        # No responses on this port
        if self.resolve_http80():
            self.http80_run_all_tests()
        else:
            self.reachable80 = False

        if self.resolve_https443():
            self.https443_run_all_tests()
        else:
            self.reachable443 = False

    def resolve_http80(self) -> bool:
        """Sets the resolved hostname, amount of redirects, and first test result"""
        try:
            ip_addr = socket.gethostbyname(self.domain)
        except Exception as error:  # pylint: disable=broad-except
            print(f"{self.domain}: {error}")
            return False

        (
            ip80,
            hostname80,
            headers_raw80,
            headers_dict80,
            counter80,
        ) = get_canonical_location(ip_addr, "http", self.domain)
        if headers_dict80["status_line"] == "<ERROR>":
            return False

        self.ip80 = ip80
        self.http80_hostname = hostname80
        self.http80_amount_of_redirects = counter80
        self.http80_add_test_result(0, headers_raw80, headers_dict80)
        return True

    def resolve_https443(self) -> bool:
        """Sets the resolved hostname, amount of redirects, and first test result"""
        try:
            ip_addr = socket.gethostbyname(self.domain)
        except Exception as error:  # pylint: disable=broad-except
            print(f"{self.domain}: {error}")
            return False

        (
            ip443,
            hostname443,
            headers_raw443,
            headers_dict443,
            counter443,
        ) = get_canonical_location(ip_addr, "https", self.domain)

        if headers_dict443["status_line"] == "<ERROR>":
            return False

        self.ip443 = ip443
        self.https443_hostname = hostname443
        self.https443_amount_of_redirects = counter443
        self.https443_add_test_result(0, headers_raw443, headers_dict443)
        return True

    def print_results(self, dump: bool = False) -> None:
        """Prints all results to stdout"""
        print("")
        print("Port 80/HTTP")
        print(f"Amount of redirects: {self.http80_amount_of_redirects}")
        print(f"Resolved hostname: {self.http80_hostname} {self.ip80}")
        print("")
        if self.reachable80:
            for case_no, result in enumerate(self.http80_test_case_results):
                print(f"Test case: {case_no}")
                if dump:
                    print(f"Raw headers:\n{repr(result[0])}")
                    print(f"Constructed headers:\n{result[1]}")

                try:
                    server = result[1]["server"]
                except KeyError:
                    server = "<EMPTY>"
                try:
                    status_line = result[1]["status_line"]
                except KeyError:
                    status_line = "<EMPTY>"

                print(f"Server: {server}")
                print(f"Status line: {status_line}")

                print("")

        print("Port 443/HTTPS")
        print(f"Amount of redirects: {self.https443_amount_of_redirects}")
        print(f"Resolved hostname: {self.https443_hostname} {self.ip443}")
        print("")
        if self.reachable443:
            for case_no, result in enumerate(self.https443_test_case_results):
                print(f"Test case: {case_no}")
                if dump:
                    print(f"Raw headers:\n{repr(result[0])}")
                    print(f"Constructed headers:\n{result[1]}")

                try:
                    server = result[1]["server"]
                except KeyError:
                    server = "<EMPTY>"
                try:
                    status_line = result[1]["status_line"]
                except KeyError:
                    status_line = "<EMPTY>"

                print(f"Server: {server}")
                print(f"Status line: {status_line}")

                print("")

    def http80_add_test_result(
        self, case_no: int, headers_raw: str, headers_dict: CaseInsensitiveDict[str]
    ) -> None:
        """Helper function to add a test case result"""
        self.http80_test_case_results.insert(case_no, (headers_raw, headers_dict))

    def https443_add_test_result(
        self, case_no: int, headers_raw: str, headers_dict: CaseInsensitiveDict[str]
    ) -> None:
        """Helper function to add a test case result"""
        self.https443_test_case_results.insert(case_no, (headers_raw, headers_dict))

    def http80_run_all_tests(self) -> None:
        """Helper function run all tests"""
        opportunistic_raw80, assumed_raw80 = run_http2_tests(
            self.ip80, "http", self.http80_hostname
        )
        self.http80_add_test_result(
            1, opportunistic_raw80, construct_header_datastructure(opportunistic_raw80)
        )
        self.http80_add_test_result(
            2, assumed_raw80, construct_header_datastructure(assumed_raw80)
        )

        test_counter = 3
        for test in SOCKET_TESTS:
            message = test.substitute(hostname=self.http80_hostname)
            headers_raw, headers_dict = send_receive_headers(
                self.ip80, "http", self.http80_hostname, message
            )
            self.http80_add_test_result(test_counter, headers_raw, headers_dict)
            test_counter += 1

    def https443_run_all_tests(self) -> None:
        """Helper function run all tests"""
        opportunistic_raw443, assumed_raw443 = run_http2_tests(
            self.ip443, "https", self.https443_hostname
        )
        self.https443_add_test_result(
            1,
            opportunistic_raw443,
            construct_header_datastructure(opportunistic_raw443),
        )
        self.https443_add_test_result(
            2, assumed_raw443, construct_header_datastructure(assumed_raw443)
        )

        test_counter = 3
        for test in SOCKET_TESTS:
            message = test.substitute(hostname=self.https443_hostname)
            headers_raw, headers_dict = send_receive_headers(
                self.ip443, "https", self.https443_hostname, message
            )
            self.https443_add_test_result(test_counter, headers_raw, headers_dict)
            test_counter += 1

    def to_tsv(self) -> str:
        """Prints all results to a tsv string"""

        line = f"{self.domain}\t{time.time()}"

        if self.reachable80:
            for result in self.http80_test_case_results:
                try:
                    server = (
                        result[1]["server"]
                        .replace("\n", "")
                        .replace("\r", "")
                        .replace("\t", "")
                    )
                except KeyError:
                    server = "<EMPTY>"
                try:
                    status_line = (
                        result[1]["status_line"]
                        .replace("\n", "")
                        .replace("\r", "")
                        .replace("\t", "")
                    )
                except KeyError:
                    status_line = "<EMPTY>"

                line += f"\t{server}\t{status_line}"
        else:
            for _ in range(len(SOCKET_TESTS) + 3):
                line += "\t\t"

        if self.reachable443:
            for result in self.https443_test_case_results:
                try:
                    server = result[1]["server"]
                except KeyError:
                    server = "<EMPTY>"
                try:
                    status_line = result[1]["status_line"]
                except KeyError:
                    status_line = "<EMPTY>"

                line += f"\t{server}\t{status_line}"
        else:
            for _ in range(len(SOCKET_TESTS) + 3):
                line += "\t\t"

        return f"{line}\n"


def run_http2_tests(ip_addr: str, protocol: str, hostname: str) -> tuple[str, str]:
    """Runs http2 tests through CURL"""
    url = protocol + "://" + hostname
    port = "80"

    if protocol == "https":
        port = "443"

    try:
        opportunistic_result = subprocess.run(
            [
                "curl",
                "--max-time",
                str(TIMEOUT),
                "--user-agent",
                USER_AGENT,
                "--resolve",
                f"{hostname}:{port}:{ip_addr}",
                "--silent",
                "--head",
                "--insecure",
                "--http2",
                url,
            ],
            stdout=subprocess.PIPE,
            check=True,
        )
        opportunistic_result_decoded = opportunistic_result.stdout.decode(
            errors="replace"
        )
    except Exception:  # pylint: disable=broad-except
        opportunistic_result_decoded = "<ERROR>"

    try:
        assumed_result = subprocess.run(
            [
                "curl",
                "--max-time",
                str(TIMEOUT),
                "--user-agent",
                USER_AGENT,
                "--resolve",
                f"{hostname}:{port}:{ip_addr}",
                "--silent",
                "--head",
                "--insecure",
                "--http2-prior-knowledge",
                url,
            ],
            stdout=subprocess.PIPE,
            check=True,
        )
        assumed_result_decoded = assumed_result.stdout.decode(errors="replace")
    except Exception:  # pylint: disable=broad-except
        assumed_result_decoded = "<ERROR>"

    return (
        opportunistic_result_decoded,
        assumed_result_decoded,
    )


def construct_header_datastructure(data: str) -> CaseInsensitiveDict[str]:
    """Constructs an ordered dict from a HTTP header as string"""
    headers: CaseInsensitiveDict[str] = CaseInsensitiveDict()
    headers_as_list = data.split("\r\n")

    if len(headers_as_list[0]) > 100:
        # edge case for headerless replies, e.g. HTTP/0.9
        headers["status_line"] = "<html>"
    else:
        headers["status_line"] = (
            headers_as_list[0].replace("\n", "").replace("\r", "").replace("\t", "")
        )

    for pair in headers_as_list[1:]:
        if len(pair) > 1:
            split_pair = pair.split(": ", 1)
            if len(split_pair) > 1:  # edge case for HTTP/2 103 early hints
                headers[split_pair[0]] = split_pair[1]
    return headers


def send_receive_headers(
    ip_addr: str, protocol: str, hostname: str, message: str
) -> tuple[str, CaseInsensitiveDict[str]]:
    """Sends and receives headers, both raw and structured"""

    byte_message = message.encode()
    headers_dict: CaseInsensitiveDict[str] = CaseInsensitiveDict()
    headers_raw: str = ""

    if protocol == "https":
        try:
            with socket.create_connection((ip_addr, 443)) as sock:
                with global_ssl_context.wrap_socket(
                    sock, server_hostname=hostname
                ) as ssock:
                    headers_raw, headers_dict = run_test(ssock, byte_message)
                    ssock.close()
                sock.close()
        except Exception as error:  # pylint: disable=broad-except
            print(f"{hostname} {ip_addr}: {error}")
            headers_dict["status_line"] = "<ERROR>"
    else:
        try:
            with socket.create_connection((ip_addr, 80)) as sock:
                headers_raw, headers_dict = run_test(sock, byte_message)
                sock.close()

        except Exception as error:  # pylint: disable=broad-except
            print(f"{hostname} {ip_addr}: {error}")
            headers_dict["status_line"] = "<ERROR>"

    return headers_raw, headers_dict


def run_test(
    sock: socket.socket, byte_message: bytes
) -> tuple[str, CaseInsensitiveDict[str]]:
    """Sends HTTP message to a socket and reads reply"""
    # print(byte_message)
    sock.sendall(byte_message)
    data_raw = sock.recv(4096)
    data = data_raw.decode(errors="replace").split("\r\n\r\n", 1)[0]
    # print(repr(data))
    return data, construct_header_datastructure(data)


def get_true_location(domain: str, headers: CaseInsensitiveDict[str]) -> str:
    """Follows a redirect to the same protocol, if actually different"""
    try:
        split_url = urlsplit(headers["location"])
        # sanity check to make sure we don't get trapped in a loop
        # print(split_url.netloc, domain)
        if split_url.netloc != domain:
            return split_url.netloc

    except (KeyError, TypeError):
        pass

    return ""


def get_canonical_location(
    ip_addr: str, protocol: str, domain: str
) -> tuple[str, str, str, CaseInsensitiveDict[str], int]:
    """Determines a canonical website domain through redirects"""
    headers_raw = ""
    headers_dict: CaseInsensitiveDict[str] = CaseInsensitiveDict()
    hostname = domain
    message = TEST_CASE_0.substitute(hostname=hostname)

    counter = 0
    while counter < 3:

        headers_raw, headers_dict = send_receive_headers(
            ip_addr, protocol, hostname, message
        )

        # We fail to connect, skip this domain
        if headers_dict["status_line"] == "<ERROR>":
            break

        true_location = get_true_location(hostname, headers_dict)

        # print(true_location)
        # If we get an empty string, we have reached the true location
        if not true_location:
            break

        hostname = true_location

        try:
            ip_addr = socket.gethostbyname(hostname)
        except Exception:  # pylint: disable=broad-except
            break

        message = TEST_CASE_0.substitute(hostname=hostname)
        counter += 1

    return ip_addr, hostname, headers_raw, headers_dict, counter


def process_domain(rank_domain: tuple[str, ...]) -> TestResults:
    """Independent function that may be multithreaded"""

    rank, domain = rank_domain
    return TestResults(int(rank), domain)


def process_results(future: concurrent.futures.Future[TestResults]) -> None:
    """Appends the results from a future to a file"""
    result = future.result()
    with open(FILENAME, "a", encoding="utf-8") as file:
        file.write(result.to_tsv())


def main() -> None:
    """Main Loop"""

    # Load dataframe of domains
    with open(sys.argv[1], newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        rank_domain_list = [tuple(row) for row in reader]

    # try:
    #     if sys.argv[2] == "--dump":
    #         DUMP = True
    # except IndexError:
    #     DUMP = False

    # process_domain((1, "darwinkel.net")).print_results()
    # sys.exit()

    with open(FILENAME, "w", encoding="utf-8") as file:
        # Append 'hello' at the end of file
        line = "domain\ttimestamp"
        for i in range((len(SOCKET_TESTS) + 3) * 2):
            line += f"\tlabel{i}\ttest{i}"
        file.write(line + "\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
        for rank_domain_tuple in rank_domain_list:
            executor.submit(process_domain, rank_domain_tuple).add_done_callback(
                process_results
            )


if __name__ == "__main__":
    main()
