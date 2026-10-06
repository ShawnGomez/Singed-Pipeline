import logging
import signal
import time
from dataclasses import dataclass

import psycopg

from singed_pipeline.indexing import POSTGRES_URL, index_section

logger = logging.getLogger("singed_pipeline.worker")

POLL_INTERVAL_SECONDS = 5
STALE_JOB_MINUTES = 15
MAX_ATTEMPTS = 5

stopping = False


@dataclass(frozen=True)
class IndexJob:
    id: str
    section_id: str
    revision: int


def request_shutdown(_signum, _frame) -> None:
    global stopping
    stopping = True


def recover_stale_jobs() -> int:
    query = """
            UPDATE document_index_jobs
            SET
                status = 'failed',
                available_at = NOW(),
                last_error = 'Worker stopped before indexing completed',
                updated_at = NOW()
            WHERE status = 'processing'
                AND updated_at < NOW() - make_interval(mins => %s)
    """

    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (STALE_JOB_MINUTES,))
            return cursor.rowcount


def claim_job() -> IndexJob | None:
    query = """
        WITH selected AS (
            SELECT id
            FROM document_index_jobs
            WHERE status IN ('pending', 'failed')
                AND available_at <= NOW()
                AND attempts < %s

            ORDER BY available_at, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )

        UPDATE document_index_jobs AS job
        SET
            status = 'processing',
            attempts = attempts + 1,
            updated_at = NOW()
        FROM selected
        WHERE job.id = selected.id
        RETURNING job.id, job.section_id, job.revision
"""

    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (MAX_ATTEMPTS,))
            row = cursor.fetchone()

    if row is None:
        return None

    return IndexJob(
        id=str(row[0]),
        section_id=str(row[1]),
        revision=row[2],
    )


def complete_job(job: IndexJob) -> bool:
    query = """
        UPDATE document_index_jobs
        SET
            status = 'completed',
            completed_at = NOW(),
            last_error = NULL,
            updated_at = NOW()
        WHERE id = %s
            AND revision = %s
            AND status = 'processing'
"""
    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (job.id, job.revision))
            return cursor.rowcount == 1


def fail_job(job: IndexJob, error: Exception) -> bool:
    # Store the exception type, not a potentially sensitive provider error message.

    safe_error = type(error).__name__

    query = """
        UPDATE document_index_jobs
        SET 
            status = 'failed',
            last_error = %s,
            available_at = NOW()
                + LEAST(3600, POWER(2, attempts)::INTEGER * 60)
                * INTERVAL '1 second',
            updated_at = NOW()
        WHERE id = %s
            AND revision = %s
            AND status = 'processing'
"""
    with psycopg.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    safe_error,
                    job.id,
                    job.revision,
                ),
            )
            return cursor.rowcount == 1

def process_job(job: IndexJob) -> None:
    try:
        index_section(job.section_id)
    except Exception as error:
        fail_job(job, error)

        logger.exception(
            "Indexing failed job_id = %s section_id = %s revision = %s",
            job.id,
            job.section_id,
            job.revision,
        )
        return

    if complete_job(job):
        logger.info(
            "Indexing completed job_id = %s section_id = %s revision %s",
            job.id,
            job.section_id,
            job.revision,
        )

    else:
        logger.info(
            "A newer revision overrode job_id= %s revision = %s",
            job.id,
            job.revision,
        )


def run_worker() -> None:
    recovered = recover_stale_jobs()
    if recovered:
        logger.warning("Recovered %s stale indexing jobs", recovered)

    while not stopping:
        try:
            job = claim_job()

            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            process_job(job)
        except Exception:
            logger.exception("Index worker loop failed")
            time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    logger.info("Document indexing worker started")
    run_worker()
    logger.info("Document indexing worker stopped")


if __name__ == "__main__":
    main()
