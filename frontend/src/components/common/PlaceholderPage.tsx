type PlaceholderPageProps = {
  eyebrow: string;
  title: string;
  message: string;
};

export function PlaceholderPage({ eyebrow, title, message }: PlaceholderPageProps) {
  return (
    <section className="placeholder-page">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{message}</p>
    </section>
  );
}
