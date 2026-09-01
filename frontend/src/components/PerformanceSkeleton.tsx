export function PerformanceSkeleton() {
  return (
    <div className="flex flex-col gap-8">
      <div>
        <div className="skeleton h-6 w-64 rounded-md" />
        <div className="skeleton mt-2 h-4 w-96 max-w-full rounded-md" />
      </div>

      <div className="skeleton h-32 w-full rounded-xl" />

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="panel p-4">
            <div className="skeleton h-3 w-20 rounded" />
            <div className="skeleton mt-2 h-7 w-16 rounded" />
            <div className="skeleton mt-2 h-3 w-24 rounded" />
          </div>
        ))}
      </div>

      <div className="panel p-6">
        <div className="skeleton h-5 w-48 rounded" />
        <div className="skeleton mt-2 h-4 w-full max-w-xl rounded" />
        <div className="skeleton mt-6 h-64 w-full rounded-lg" />
      </div>
    </div>
  );
}
